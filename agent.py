#!/usr/bin/env python3
"""AI Agent for email task extraction using local AI tools."""

import asyncio
import datetime
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import List, Optional

import tensorflow_probability as tfp
from src.task_card_generator.html_generator import create_task_html_image
from src.task_card_generator.printer import print_to_thermal_printer
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

TOKEN_PATH = 'token.json'

def get_gmail_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
    return creds
from src.task_card_generator.ai_client import analyze_emails_for_tasks, parse_task_analysis
from dotenv import load_dotenv
from pydantic import BaseModel

from src.database.task_db import TaskDatabase, TaskRecord

tfd = tfp.distributions

# Load environment variables
load_dotenv()


class Task(BaseModel):
    """Task model for extracted email tasks."""

    name: str
    priority: int  # 1 for high, 2 for medium, 3 for low
    due_date: str  # ISO format date string




async def extract_email_tasks(
    user_email: Optional[str] = None,
    ) -> dict:
    """
    Extract tasks from Gmail emails using local AI endpoint.

    Args:
        user_email: User email for context (optional)

    Returns:
        ImportantTasks object containing extracted tasks and summary
    """

    # Try to fetch emails using Gmail API
    emails_content = ""
    try:
        creds = get_gmail_credentials()
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=20).execute()
        messages = results.get('messages', [])
        formatted_emails = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            snippet = msg_data.get('snippet', '')
            headers = msg_data.get('payload', {}).get('headers', [])
            sender = None
            subject = None
            for h in headers:
                if h['name'].lower() == 'from':
                    sender = h['value']
                if h['name'].lower() == 'subject':
                    subject = h['value']
            formatted_email = f"From: {sender if sender else 'Unknown'}\nSubject: {subject if subject else ''}\nBody: {snippet}"
            formatted_emails.append(formatted_email)
        emails_content = '\n---\n'.join(formatted_emails)
    except Exception as e:
        print(f"[WARN] Could not fetch emails via Gmail API: {e}\nUsing placeholder emails.")
        emails_content = """
        From: alice@example.com\nSubject: Project Update\nBody: Please review the attached report by Friday.\n---\nFrom: bob@example.com\nSubject: Team Meeting\nBody: Don't forget the team meeting tomorrow at 10am.\n---\nFrom: promo@store.com\nSubject: Special Offer\nBody: This is a promotional offer, ignore.\n        """

    # Print the email content
    print("\n--- EMAIL CONTENT ---")
    print(emails_content)
    print("--- END EMAIL CONTENT ---\n")

    # Send to local AI endpoint
    ai_response = analyze_emails_for_tasks(emails_content)
    print("\n[DEBUG] Raw AI response:\n", ai_response, "\n")
    tasks = parse_task_analysis(ai_response)

    # Compose summary
    summary = f"Extracted {len(tasks)} tasks from emails."

    # Return both the full dicts and the summary
    return {"tasks": tasks, "summary": summary}


async def main():
    """Main entry point for the email task extraction agent."""
    print("=" * 50)
    print("EMAIL TASK EXTRACTION AGENT")
    print("Powered by My OWN Tools >:)")
    print("=" * 50)

    # Initialize database
    db = TaskDatabase()
    db._create_tables()

    # Get user email from environment or ask
    user_email = os.getenv("USER_EMAIL")
    if not user_email:
        user_email = input("\nEnter your email address: ")

    print(f"\n📧 Analyzing emails for: {user_email}")
    print("🔄 Processing...")

    # Extract tasks
    result = await extract_email_tasks(user_email=user_email)
    print(result)
    tasks = result["tasks"]
    summary = result["summary"]
    if tasks:
        print(f"\n✅ Found {len(tasks)} tasks")
        print("\n📊 SUMMARY:")
        print(f"   {summary}")

        print("\n📋 EXTRACTED TASKS:")
        priority_map = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}

        # Process and store tasks
        new_tasks = []
        duplicate_tasks = []

        for i, task in enumerate(tasks, 1):
            print(f"\n{i}. {task['title']}")
            print(f"   Priority: {priority_map.get(task['priority'], '❓ UNKNOWN')}")
            print(f"   Due: {task.get('deadline', '')}")

            # Check for duplicates
            is_duplicate = False
            similar_tasks = db.find_similar_tasks(task['title'])
            if (
                similar_tasks
                and len(similar_tasks) > 0
                and similar_tasks[0].similarity_distance < 0.1
            ):
                is_duplicate = True
                duplicate_tasks.append(task)
                continue

            # Add new task to database
            db_task = TaskRecord(
                name=task['title'],
                priority={"HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(task['priority'], 2),
                due_date=task.get('deadline', ''),
                created_at=datetime.datetime.now().isoformat(),
            )
            db.add_task(db_task)
            new_tasks.append(task)

            # Print receipt for the new task (pass full dict)
            image_path = create_task_html_image(task)
            if image_path:
                print_to_thermal_printer(image_path)

        # Print summary of database operations
        if new_tasks:
            print(f"\n💾 Saved {len(new_tasks)} new tasks to database")
        if duplicate_tasks:
            print(
                f"\n🔁 Found {len(duplicate_tasks)} duplicate tasks that were not saved:"
            )
            for task in duplicate_tasks:
                print(f"   - {task['title']}")
    else:
        print("\n❌ No actionable tasks found in recent emails")

    print("\n" + "=" * 50)
    # Close database connection
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
