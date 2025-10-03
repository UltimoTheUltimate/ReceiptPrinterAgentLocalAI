
"""Ollama API client for task generation."""

import os
import ollama


def is_promotional_email(email_content):
    """Ask the model if the email is promotional/automated. Returns True if promotional, else False."""
    prompt = f"""
    Is the following email promotional, marketing, automated, a notification, password reset or sent by a system/robot? Only answer YES or NO.

    Email:
    {email_content}

    Answer YES if the email is promotional, marketing, automated, notification, password reset or sent by a system/robot (including noreply, no-reply, notification, mailer, robot, bot, do-not-reply, system, admin, support, update, service, info@, etc.).
    Answer NO if the email is from a real person and requires a genuine response or action.

    Only output YES or NO.
    """
    try:
        response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 1000}
        )
        raw_output = response['message']['content']
        print(f"[DEBUG] Raw promo check output: {repr(raw_output)}")
        import re
        # Remove <think>...</think> blocks
        no_think = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
        # Remove any remaining tags
        cleaned = re.sub(r'<.*?>', '', no_think, flags=re.DOTALL).strip().upper()
        cleaned = cleaned.replace('\n', '').replace('\r', '').replace(' ', '')
        print(f"[DEBUG] Cleaned promo check answer: {cleaned}")
        has_yes = 'YES' in cleaned
        has_no = 'NO' in cleaned
        if has_yes and not has_no:
            return True
        elif has_no and not has_yes:
            return False
        else:
            print(f"[WARN] Ambiguous promo check output: {repr(raw_output)}. Treating as NOT promotional.")
            return False
    except Exception as e:
        print(f"[WARN] Promo check failed: {e}")
        return False

def get_task_from_ai(task_description):
    prompt = f"""
    You are an expert assistant. Convert the following task description into a structured JSON object for downstream processing.

    Task Description: {task_description}

    Please provide a JSON object with the following fields:
    {{
        "title": "A short, clear task title (max 25 characters)",
        "priority": "HIGH, MEDIUM, or LOW",
        "deadline": "Any mentioned deadline or 'None'",
        "reason": "Why this needs attention",
        "from": "Who requested or sent the task (if known, else 'Unknown')"
    }}

    Return ONLY the JSON object. Do not include any explanation, extra text and stick to the formatting. your response will be parsed as JSON by a script.
    """

    try:
        response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 10000}
        )
        ai_content = response['message']['content']
        print(f"[DEBUG] Raw AI output from model:\n{ai_content}\n")
        # Remove <think>...</think> and stray </think> tags
        import re
        cleaned_content = re.sub(r'<think>.*?</think>', '', ai_content, flags=re.DOTALL)
        cleaned_content = cleaned_content.replace('</think>', '').strip()
        # Remove Markdown code block markers
        cleaned_content = re.sub(r'```(?:json)?', '', cleaned_content).strip()
        # Try to parse as JSON first
        import json
        try:
            data = json.loads(cleaned_content)
            # If it's a list, take the first task
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                task = data[0]
                # Normalize keys for compatibility
                return {
                    "title": task.get("title", "TASK"),
                    "priority": task.get("priority", "MEDIUM"),
                    "deadline": task.get("deadline", "None"),
                    "reason": task.get("reason", "No reason provided"),
                    "from": task.get("from", "Unknown"),
                }
            elif isinstance(data, dict):
                return {
                    "title": data.get("title", "TASK"),
                    "priority": data.get("priority", "MEDIUM"),
                    "deadline": data.get("deadline", "None"),
                    "reason": data.get("reason", "No reason provided"),
                    "from": data.get("from", "Unknown"),
                }
        except Exception:
            pass
        # Fallback to legacy parsing
        return parse_ai_response(ai_content)
    except Exception as e:
        return {"title": f"Error: {str(e)}", "priority": "MEDIUM"}


def analyze_emails_for_tasks(emails_content):
    """Analyze Gmail emails to identify actionable tasks."""
    prompt = f"""
    You are an email assistant. Only include emails that require a real response or action from you, sent by real people (not automated systems).

    Emails:
    {emails_content}

    INCLUDE:
    - Meeting invites, requests for info, project updates needing acknowledgment, collaboration, time-sensitive content.

    EXCLUDE:
    - Promotional, marketing, newsletters, notifications, system alerts, login codes, password resets, security alerts, and any email from senders like noreply, no-reply, notification, mailer, robot, bot, do-not-reply, system, admin, support, update, service, info@, etc.

    For each actionable email, create a task with:
    - title: What needs to be done
    - from: Who sent it
    - priority: HIGH, MEDIUM, or LOW
    - deadline: Any mentioned deadline or "None"
    - reason: Why this needs attention

    Return ONLY a JSON array of tasks, no explanation or extra text. Example:
    [{{"title": "...", "from": "...", "priority": "...", "deadline": "...", "reason": "..."}}]
    If no action is needed, return []
    """
        

    try:
        response = ollama.chat(
            model="deepseek-r1:7b",
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
    cleaned_response = cleaned_response.replace('</think>', '').strip()
    # Remove Markdown code block markers
    cleaned_response = re.sub(r'```(?:json)?', '', cleaned_response).strip()


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
        # List of patterns that indicate an automated sender
        automated_patterns = [
            'noreply', 'no-reply', 'notification', 'mailer', 'robot', 'bot',
            'do-not-reply', 'system', 'admin', 'support', 'update', 'service', 'info@',
            'donotreply', 'auto', 'automated', 'alerts', 'reminder', 'news', 'digest', 'newsletter', 'bounce', 'daemon', 'postmaster'
        ]
        import re
        for task in tasks:
            if isinstance(task, dict) and "title" in task:
                # Support alternative key names for sender and reason
                sender = task.get("from") or task.get("sender") or task.get("email") or "Unknown"
                reason = task.get("reason") or task.get("why") or task.get("description") or "No reason provided"
                # Filter out tasks with automated senders
                sender_lower = str(sender).strip().lower()
                if any(pat in sender_lower for pat in automated_patterns):
                    continue
                # Also filter out if sender looks like an email address but is missing a real name
                if re.match(r"^[^@]+@(noreply|no-reply|notifications?|mailer|robot|bot|do-not-reply|system|admin|support|update|service|info|donotreply|auto|automated|alerts|reminder|news|digest|newsletter|bounce|daemon|postmaster)\.", sender_lower):
                    continue
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
