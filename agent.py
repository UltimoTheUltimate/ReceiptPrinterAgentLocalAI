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
    db=None
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
        # Set number of days ago
        days_ago = int(os.getenv('EMAIL_DAYS_AGO', '7'))  # Default to 7 days
        after_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y/%m/%d')
        query = f'after:{after_date.replace("/", "/")}'
        results = service.users().messages().list(userId='me', q=query, maxResults=20).execute()
        messages = results.get('messages', [])
        all_tasks = []
        for msg in messages:
            msg_id = msg['id']
            # Check if this email has already been processed
            if hasattr(db, 'email_exists'):
                if db.email_exists(msg_id):
                    print(f"[SKIP] Email {msg_id} already processed.")
                    continue
            msg_data = service.users().messages().get(userId='me', id=msg_id).execute()
            # Extract full body from Gmail message payload
            payload = msg_data.get('payload', {})
            body = ''
            import base64
            if 'parts' in payload:
                for part in payload['parts']:
                    if 'data' in part.get('body', {}):
                        decoded = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        body = decoded
                        break
            elif 'body' in payload and 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
            else:
                body = msg_data.get('snippet', '')

            # Strip HTML/XML tags from body
            try:
                from bs4 import BeautifulSoup
                body = BeautifulSoup(body, "html.parser").get_text()
            except ImportError:
                import re
                body = re.sub(r'<[^>]+>', '', body)

            # Trim and clean the email body for small models
            # Remove quoted replies (lines starting with '>'), common signature delimiters, and limit length
            lines = body.splitlines()
            cleaned_lines = []
            for line in lines:
                # Stop at signature delimiter or reply header
                if line.strip().startswith('--') or line.strip().startswith('On ') and 'wrote:' in line:
                    break
                if line.strip().startswith('>'):
                    continue
                cleaned_lines.append(line)
            trimmed_body = '\n'.join(cleaned_lines).strip()
            # Limit to first 1000 characters
            if len(trimmed_body) > 1000:
                trimmed_body = trimmed_body[:1000] + '...'
            body = trimmed_body

            headers = payload.get('headers', [])
            sender = None
            subject = None
            date_str = None
            for h in headers:
                if h['name'].lower() == 'from':
                    sender = h['value']
                if h['name'].lower() == 'subject':
                    subject = h['value']
                if h['name'].lower() == 'date':
                    date_str = h['value']
            formatted_email = f"From: {sender if sender else 'Unknown'}\nSubject: {subject if subject else ''}\nDate: {date_str if date_str else ''}\nBody: {body}"
            print("\n--- EMAIL CONTENT ---")
            print(formatted_email)
            print("--- END EMAIL CONTENT ---\n")
            # Step 1: Ask if the email is promotional/automated
            from src.task_card_generator.ai_client import is_promotional_email
            print("[DEBUG] Checking if email is promotional/automated...")
            promo_result = is_promotional_email(formatted_email)
            print(f"[DEBUG] is_promotional_email returned: {promo_result}")
            if promo_result:
                print("[SKIP] Email detected as promotional/automated. Skipping task extraction.")
                continue
            # Step 2: Prompt AI for this email only if not promotional
            ai_response = analyze_emails_for_tasks(formatted_email)
            print("\n[DEBUG] Raw AI response:\n", ai_response, "\n")
            tasks = parse_task_analysis(ai_response)
            all_tasks.extend(tasks)
            # Mark this email as processed
            if hasattr(db, 'mark_email_processed'):
                db.mark_email_processed(msg_id)
        summary = f"Extracted {len(all_tasks)} tasks from emails after {after_date}."
        return {"tasks": all_tasks, "summary": summary}
    except Exception as e:
        print(f"[WARN] Could not fetch emails via Gmail API: {e}\nUsing placeholder emails.")
        emails = [
            "From: alice@example.com\nSubject: Project Update\nDate: 2025-09-18\nBody: Please review the attached report by Friday.",
            "From: bob@example.com\nSubject: Team Meeting\nDate: 2025-09-18\nBody: Don't forget the team meeting tomorrow at 10am.",
            "From: promo@store.com\nSubject: Special Offer\nDate: 2025-09-18\nBody: This is a promotional offer, ignore."
        ]
        all_tasks = []
        for formatted_email in emails:
            print("\n--- EMAIL CONTENT ---")
            print(formatted_email)
            print("--- END EMAIL CONTENT ---\n")
            ai_response = analyze_emails_for_tasks(formatted_email)
            print("\n[DEBUG] Raw AI response:\n", ai_response, "\n")
            tasks = parse_task_analysis(ai_response)
            all_tasks.extend(tasks)
        summary = f"Extracted {len(all_tasks)} tasks from placeholder emails."
        return {"tasks": all_tasks, "summary": summary}


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
    result = await extract_email_tasks(user_email=user_email, db=db)
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

            # Improved duplicate detection
            normalized_title = task['title'].strip().lower()
            task_priority = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(task['priority'], 2)
            task_deadline = task.get('deadline', '').strip()
            is_duplicate = False
            similar_tasks = db.find_similar_tasks(normalized_title)
            if similar_tasks and len(similar_tasks) > 0:
                for sim_task in similar_tasks:
                    sim_title = getattr(sim_task, 'name', '').strip().lower()
                    sim_priority = getattr(sim_task, 'priority', 2)
                    sim_deadline = getattr(sim_task, 'due_date', '').strip()
                    sim_distance = getattr(sim_task, 'similarity_distance', None)
                    # Stricter threshold and compare more fields
                    if (
                        sim_distance is not None and sim_distance < 0.05 and
                        sim_title == normalized_title and
                        sim_priority == task_priority and
                        sim_deadline == task_deadline
                    ):
                        is_duplicate = True
                        duplicate_tasks.append(task)
                        break
            if is_duplicate:
                continue

            # Add new task to database
            db_task = TaskRecord(
                name=task['title'],
                priority=task_priority,
                due_date=task_deadline,
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
