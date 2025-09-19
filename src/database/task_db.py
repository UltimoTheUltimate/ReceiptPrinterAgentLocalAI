"""Task database with embeddings using Turso and libsql."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

import libsql_experimental as libsql
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class TaskRecord:
    """Task record with embedding."""

    id: Optional[int] = None
    name: str = ""
    priority: int = 2
    due_date: str = ""
    created_at: str = ""
    embedding: Optional[List[float]] = None
    email_context: Optional[str] = None
    similarity_distance: Optional[float] = (
        None  # Cosine distance for similarity searches
    )


class TaskDatabase:
    """Manage tasks with embeddings in Turso database."""

    def __init__(self, db_url: Optional[str] = None, auth_token: Optional[str] = None):
        """Initialize database connection."""
        self.db_url = db_url or os.getenv("TURSO_DATABASE_URL")
        self.auth_token = auth_token or os.getenv("TURSO_AUTH_TOKEN")
        # Use local AI instance for embeddings
        self.openai = OpenAI(
            base_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama ignores the key, but OpenAI client requires it
        )
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.embedding_dimensions = 768  # Adjust to your local model's output size

        # Connect to database
        if self.db_url and self.auth_token:
            # Remote Turso database
            self.conn = libsql.connect(self.db_url, auth_token=self.auth_token)
        else:
            # Local SQLite database
            self.conn = libsql.connect("tasks.db")

    def _create_tables(self):
        """Create tasks table with embeddings if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                priority INTEGER NOT NULL,
                due_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                email_context TEXT,
                embedding F32_BLOB({self.embedding_dimensions})
            )
        """)

        # Create vector index for similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS tasks_embedding_idx 
            ON tasks(libsql_vector_idx(embedding))
        """)

        self.conn.commit()
        cursor.close()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using local AI instance."""
        response = self.openai.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def add_task(self, task: Any, email_context: Optional[str] = None) -> TaskRecord:
        """Add a task to the database with embeddings."""
        # Generate embedding from task name and context
        embedding_text = f"{task.name}"
        if email_context:
            embedding_text += f" Context: {email_context}"

        embedding = self.generate_embedding(embedding_text)

        # Convert embedding to vector format
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO tasks (name, priority, due_date, created_at, email_context, embedding)
            VALUES (?, ?, ?, ?, ?, vector32(?))
        """,
            (
                task.name,
                task.priority,
                task.due_date,
                datetime.now().isoformat(),
                email_context,
                embedding_str,
            ),
        )

        task_id = cursor.lastrowid
        self.conn.commit()
        cursor.close()

        return TaskRecord(
            id=task_id,
            name=task.name,
            priority=task.priority,
            due_date=task.due_date,
            created_at=datetime.now().isoformat(),
            embedding=embedding,
            email_context=email_context,
        )

    def find_similar_tasks(self, query: str, limit: int = 5) -> List[TaskRecord]:
        """Find similar tasks based on embedding similarity."""
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 
                t.id, t.name, t.priority, t.due_date, t.created_at, 
                t.email_context,
                vector_distance_cos(t.embedding, vector32(?)) as distance
            FROM vector_top_k('tasks_embedding_idx', vector32(?), ?) AS v
            JOIN tasks t ON t.rowid = v.id
            ORDER BY distance ASC
        """,
            (embedding_str, embedding_str, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                TaskRecord(
                    id=row[0],
                    name=row[1],
                    priority=row[2],
                    due_date=row[3],
                    created_at=row[4],
                    email_context=row[5],
                    similarity_distance=row[6],  # Include the distance from the query
                )
            )

        cursor.close()
        return results

    def get_recent_tasks(self, limit: int = 10) -> List[TaskRecord]:
        """Get most recent tasks."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, priority, due_date, created_at, email_context
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                TaskRecord(
                    id=row[0],
                    name=row[1],
                    priority=row[2],
                    due_date=row[3],
                    created_at=row[4],
                    email_context=row[5],
                )
            )

        cursor.close()
        return results

    def close(self):
        """Close database connection."""
        self.conn.close()
