"""OpenAI API client for task generation."""

import os

from openai import OpenAI


def get_task_from_ai(task_description):
    # Use Ollama's OpenAI-compatible API endpoint
    client = OpenAI(
        base_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # Ollama ignores the key, but OpenAI client requires it
    )

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
        response = client.chat.completions.create(
            model="deepseek-r1:14b",  # Use DeepSeek 14B model (Ollama)
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_emails_for_tasks(emails_content):
    """Analyze Gmail emails to identify actionable tasks."""
    # Use Ollama's OpenAI-compatible API endpoint
    client = OpenAI(
        base_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # Ollama ignores the key, but OpenAI client requires it
    )

    prompt = f"""
    You are an email assistant. Look at these emails and find ANY that might need a response, follow-up, or action.

    Emails:
    {emails_content}

    Be LIBERAL in what you consider actionable. Include:
    - Meeting invites or webinars (even if promotional)
    - Requests for information or responses
    - Business opportunities
    - Project updates that need acknowledgment
    - Anything from real people (not just automated systems)
    - Time-sensitive content
    - Collaboration requests

    For each email that needs ANY kind of action, create a task with:
    - title: What needs to be done
    - from: Who sent it
    - priority: HIGH, MEDIUM, or LOW
    - deadline: Any mentioned deadline or "None"
    - reason: Why this needs attention

    Return a JSON array of tasks. Be generous - when in doubt, include it.
    Format: [{{"title": "...", "from": "...", "priority": "...", "deadline": "...", "reason": "..."}}]

    If truly nothing needs action, return: []
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-r1:14b",  # Use DeepSeek 14B model (Ollama)
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content
            refusal = getattr(message, "refusal", None)

            if content is None and refusal:
                # Try with a simpler prompt without strict JSON formatting
                simple_prompt = f"""
                Look at these emails and identify any that need a response or action:

                {emails_content[:2000]}

                For each actionable email, provide a JSON object with:
                - "title": What needs to be done in under 25 characters
                - "from": Who sent it
                - "priority": HIGH, MEDIUM, or LOW
                - "deadline": Any mentioned deadline or "None"
                - "reason": Why this needs attention

                Return a JSON array of such objects in exactly this format. Example:
                [
                  {{
                    "title": "Reply to project update",
                    "from": "alice@example.com",
                    "priority": "HIGH",
                    "deadline": "2025-09-20",
                    "reason": "Project update requires response"
                  }}
                ]

                If truly nothing needs action, return: []
                """

                fallback_response = client.chat.completions.create(
                    model="deepseek-r1:14b",  # Use DeepSeek 14B model (Ollama)
                    messages=[{"role": "user", "content": simple_prompt}],
                    max_tokens=10000,
                    temperature=0.1,
                )

                fallback_content = fallback_response.choices[0].message.content
                print("AI reply:", fallback_content)
                return fallback_content or "Error: OpenAI returned None content"

            if content is None:
                return "Error: OpenAI returned None content"

            return content
        else:
            return "Error: No response choices from OpenAI"

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg


def parse_task_analysis(analysis_response):
    """Parse the AI analysis response into structured task data."""
    if not analysis_response or analysis_response.startswith("Error:"):
        return []

    try:
        import json


        # Parse the JSON response
        response_data = json.loads(analysis_response)

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
