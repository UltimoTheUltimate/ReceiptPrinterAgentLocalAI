"""HTML to image generation for task cards."""

import tempfile
from datetime import datetime

try:
    import imgkit
    IMGKIT_AVAILABLE = True
except ImportError:
    IMGKIT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


def create_task_html(task):
    """Create HTML content for task card with ticket-style design."""
    # Use the due_date from task if it's a Task object, otherwise use current date
    from datetime import datetime as dt
    if hasattr(task, 'due_date') and task.due_date:
        try:
            due_date_obj = dt.fromisoformat(str(task.due_date).replace('Z', '+00:00'))
        except Exception:
            due_date_obj = datetime.now()
    else:
        due_date_obj = datetime.now()
    due_date_text = due_date_obj.strftime('%B %d')
    day = due_date_obj.day
    
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    
    due_date_text = f"{due_date_text}{suffix}"
    
    # Priority indicator - handle both Task object and dict
    if hasattr(task, 'priority'):
        # Task object with numeric priority
        priority_map = {1: ("⚡ ⚡ ⚡", "HIGH PRIORITY"),
                       2: ("⚡ ⚡", "MEDIUM PRIORITY"),
                       3: ("⚡", "LOW PRIORITY")}
        priority_dots, priority_text = priority_map.get(task.priority, ("⚡ ⚡", "MEDIUM PRIORITY"))
    else:
        # Dict format with string priority
        if task["priority"].upper() == "HIGH":
            priority_dots = "⚡ ⚡ ⚡"
            priority_text = "HIGH PRIORITY"
        elif task["priority"].upper() == "MEDIUM":
            priority_dots = "⚡ ⚡"
            priority_text = "MEDIUM PRIORITY"
        else:
            priority_dots = "⚡"
            priority_text = "LOW PRIORITY"
    
    # Null-safe extraction of all possible fields
    title = getattr(task, 'name', None) or getattr(task, 'title', None) or task.get('title', '') if isinstance(task, dict) else ''
    priority = priority_text
    due_date = due_date_text
    reason = getattr(task, 'reason', None) or (task.get('reason', '') if isinstance(task, dict) else '')
    sender = getattr(task, 'from', None) or (task.get('from', '') if isinstance(task, dict) else '')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Apple Color Emoji', 'Noto Color Emoji', Arial, sans-serif;
                background-color: white;
                width: 576px;
                padding: 0;
                margin: 0;
            }}
            .ticket-container {{
                background: white;
                padding: 10px 5px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 5px;
            }}
            .ticket-label {{
                font-size: 24px;
                font-weight: bold;
                letter-spacing: 4px;
                color: #000;
                margin-bottom: 16px;
            }}
            .priority-dots {{
                font-size: 48px;
                font-weight: bold;
                margin-top: 4px;
                color: #000;
                font-family: 'Segoe UI Emoji', 'Segoe UI Symbol', 'Apple Color Emoji', 'Noto Color Emoji', 'Segoe UI', Arial, sans-serif;
            }}
            .perforation {{
                background: repeating-linear-gradient(
                    to right,
                    #000 0,
                    #000 6px,
                    transparent 6px,
                    transparent 12px
                );
                height: 3px;
                margin: 5px 0;
            }}
            .task-title {{
                text-align: center;
                padding: 10px 0;
            }}
            .task-title h1 {{
                font-size: 48px;
                font-weight: bold;
                line-height: 1.2;
                color: #000;
                word-wrap: break-word;
                max-width: 100%;
                overflow-wrap: break-word;
                hyphens: auto;
                margin: 0;
                padding: 0 10px;
            }}
            .dashed-line {{
                border-top: 3px dashed #666;
                margin: 5px 0;
            }}
            .due-date {{
                text-align: center;
            }}
            .due-date-text {{
                font-size: 32px;
                font-weight: bold;
                color: #000;
            }}
            .info-section {{
                margin: 10px 0;
                padding: 0 10px;
                font-size: 20px;
                color: #222;
            }}
            .info-label {{
                font-weight: bold;
                color: #444;
            }}
            .bottom-perforation {{
                margin-top: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="ticket-container">
            <!-- Lightning Bolts Only -->
            <div class="header">
                <div class="priority-dots">{priority_dots}</div>
            </div>
            <!-- Task Title -->
            <div class="task-title">
                <h1>{getattr(task, 'name', None) or getattr(task, 'title', None) or (task.get('title', '') if isinstance(task, dict) else '')}</h1>
            </div>
            <!-- Dashed Separator -->
            <div class="dashed-line"></div>
            <!-- Due Date Section -->
            <div class="due-date">
                <div class="due-date-text">{due_date_text}</div>
            </div>
            <!-- Info Section: Sender and Reason -->
            <div class="info-section">
                <span class="info-label">From:</span> {sender if sender else 'Unknown'}<br>
                <span class="info-label">Reason:</span> {reason if reason else 'No reason provided'}
            </div>
            <!-- Bottom Perforation -->
            <div class="perforation bottom-perforation"></div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def html_to_image_imgkit(html_content):
    """Convert HTML to image using imgkit (requires wkhtmltopdf)."""
    if not IMGKIT_AVAILABLE:
        print("imgkit not available - cannot convert HTML to image")
        return None
    
    try:
        # Create temporary image file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img.close()
        
        # Configure options for thermal printer size
        options = {
            'width': 576,  # 72mm thermal printer width
            'disable-smart-width': '',
            'encoding': 'UTF-8',
            'disable-local-file-access': '',
            'crop-w': 576,  # Crop to exact width
        }
        
        # Try to configure wkhtmltopdf path for Windows
        config = None
        import os
        possible_paths = [
            r'C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe',
            r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltoimage.exe',
            r'C:\wkhtmltopdf\bin\wkhtmltoimage.exe'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config = imgkit.config(wkhtmltoimage=path)
                break
        
        # Convert HTML to image
        imgkit.from_string(html_content, temp_img.name, options=options, config=config)
        
        print(f"HTML converted to image: {temp_img.name}")
        return temp_img.name
        
    except Exception as e:
        print(f"Error converting HTML to image with imgkit: {str(e)}")
        return None


def html_to_image_selenium(html_content):
    """Convert HTML to image using Selenium (requires Chrome/ChromeDriver)."""
    if not SELENIUM_AVAILABLE:
        print("Selenium not available - cannot convert HTML to image")
        return None
    
    try:
        # Create temporary HTML file
        temp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
        temp_html.write(html_content)
        temp_html.close()
        
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=576,800')  # Tall enough to capture content
        
        # Create webdriver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load HTML file
        driver.get(f'file://{temp_html.name}')
        
        # Wait for page to load
        time.sleep(1)
        
        # Get the ticket container element
        ticket_element = driver.find_element(By.CLASS_NAME, "ticket-container")
        
        # Create temporary image file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img.close()
        
        # Take screenshot of just the ticket element
        ticket_element.screenshot(temp_img.name)
        driver.quit()
        
        print(f"HTML converted to image: {temp_img.name}")
        return temp_img.name
        
    except Exception as e:
        print(f"Error converting HTML to image with Selenium: {str(e)}")
        return None


def create_task_html_image(task_data):
    """Create task card image from HTML using available method."""
    html_content = create_task_html(task_data)
    
    # Try imgkit first (faster), then Selenium
    image_path = html_to_image_imgkit(html_content)
    if image_path is None:
        image_path = html_to_image_selenium(html_content)
    
    return image_path