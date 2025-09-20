"""Ollama API client for task generation."""

import os
import ollama


def get_task_from_ai(task_description):
    prompt = f"""
    Convert this task description into a clear, concise task name:

    Task Description: {task_description}

    Please provide:
    1. A short, clear task title (max 25 characters)
    2. Priority level (HIGH, MEDIUM, LOW)

    Format your response exactly like this:
    TITLE: [task title]
    PRIORITY: [priority level]
    """

    try:
        response = ollama.chat(
            model="deepseek-r1:14b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 10000}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_emails_for_tasks(emails_content):
    """Analyze Gmail emails to identify actionable tasks."""
    prompt = f"""
    You are an email assistant. Look at this email and identify one or more tasks that require action.

    Emails:
    {emails_content}

    You may Include:
    - Meeting invites or webinars
    - Requests for information or responses
    - Project updates that need acknowledgment
    - Anything from real people (not just automated systems)
    - Time-sensitive content
    - Collaboration requests

    Do NOT include:
    - Purely promotional emails with no call to action
    - Newsletters or informational emails that don't require a response
    - Social media notifications
    - System alerts or automated messages with no action needed
    - Any sales or marketing emails
    - Promotional, marketing, and automated system emails (e.g., newsletters, offers, sales, login codes, notifications, Quillbot, Ground News, Heavyocity, etc.)

    Only include emails that require a genuine response or action from you, especially those sent by real people or related to your work, projects, or meetings.

    For each email that needs ANY kind of action, create a task with:
    - title: What needs to be done
    - from: Who sent it
    - priority: HIGH, MEDIUM, or LOW
    - deadline: Any mentioned deadline or "None"
    - reason: Why this needs attention

    Return ONLY the JSON array. NO explanation and NO extra text other than the JSON array.
    Return ONLY the tasks in ONLY the following format. : [{{"title": "...", "from": "...", "priority": "...", "deadline": "...", "reason": "..."}}]
    Your full response will be parsed as JSON by a script.
    If the email does not require attention simply return []
    """
        

    try:
        response = ollama.chat(
            model="deepseek-r1:14b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 10000}
        )
        return response['message']['content']
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg


def parse_task_analysis(analysis_response):
    """Parse the AI analysis response into structured task data."""
    if not analysis_response or analysis_response.startswith("Error:"):
        return []

    # Remove <think>...</think> sections if present
    import re
    cleaned_response = re.sub(r'<think>.*?</think>', '', analysis_response, flags=re.DOTALL)

    try:
        import json

        # Parse the JSON response
        response_data = json.loads(cleaned_response)

        # Handle different JSON structures
        if isinstance(response_data, list):
            # Direct array of tasks
            tasks = response_data
        elif isinstance(response_data, dict):
            # Check for common wrappers
            if "tasks" in response_data:
                tasks = response_data["tasks"]
            elif "data" in response_data:
                tasks = response_data["data"]
            elif "items" in response_data:
                tasks = response_data["items"]
            else:
                # Assume the dict itself is a single task
                tasks = [response_data]
        else:
            return []

        # Validate and clean up tasks
        valid_tasks = []
        for task in tasks:
            if isinstance(task, dict) and "title" in task:
                # Support alternative key names for sender and reason
                sender = task.get("from") or task.get("sender") or task.get("email") or "Unknown"
                reason = task.get("reason") or task.get("why") or task.get("description") or "No reason provided"
                clean_task = {
                    "title": task.get("title", "").strip()[:50],  # Limit title length
                    "from": str(sender).strip()[:30],  # Limit sender length
                    "priority": task.get("priority", "MEDIUM").strip().upper(),
                    "deadline": task.get("deadline", "None").strip()[:20],  # Limit deadline length
                    "reason": str(reason).strip()[:100],  # Limit reason length
                }
                # Only add if we have a meaningful task title
                if clean_task["title"] and len(clean_task["title"]) > 3:
                    # Validate priority
                    if clean_task["priority"] not in ["HIGH", "MEDIUM", "LOW"]:
                        clean_task["priority"] = "MEDIUM"
                    valid_tasks.append(clean_task)
        return valid_tasks

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Error parsing task analysis: {e}")
        return []


def parse_ai_response(response):
    """Parse AI response to extract task components."""
    task_data = {"title": "TASK", "priority": "MEDIUM"}

    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("TITLE:"):
            task_data["title"] = line.replace("TITLE:", "").strip()
        elif line.startswith("PRIORITY:"):
            task_data["priority"] = line.replace("PRIORITY:", "").strip()

    return task_data
