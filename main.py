#!/usr/bin/env python3
"""Simple CLI for AI-powered task extraction and printing."""

import os
from dotenv import load_dotenv
from src.task_card_generator import (
    create_task_image,
    create_task_pdf,
    get_task_from_ai,
    parse_ai_response,
    print_to_thermal_printer,
)
from src.task_card_generator.html_generator import create_task_html_image

# Load environment variables
load_dotenv()


def main():
    """Main CLI entry point for task generation."""
    print("=" * 50)
    print("AI TASK GENERATOR")
    print("Turn your thoughts into actionable tasks")
    print("=" * 50)

    # Get user task description
    print("\nDescribe the task you need to complete:")
    task_description = input(">> ")

    if not task_description.strip():
        print("ERROR: Please enter a valid task description!")
        return

    # Get AI response
    print("\n🤖 Processing with AI...")
    ai_response = get_task_from_ai(task_description)

    print(f"\n[DEBUG] Raw AI response:\n{ai_response}\n")

    if not ai_response or ai_response.startswith("Error"):
        print(f"ERROR: {ai_response or 'Failed to get response'}")
        return

    # Parse AI response
    task_data = parse_ai_response(ai_response)

    print(f"\n✅ Generated Task: {task_data['title']}")
    print(f"📌 Priority: {task_data['priority']}")
    print(f"📅 Due: {task_data.get('due_date', 'Today')}")

    # Create outputs
    print("\n📄 Creating outputs...")
    
    # Create PDF for viewing
    pdf_path = create_task_pdf(task_data)
    if pdf_path:
        print(f"   PDF saved: {pdf_path}")

    # Create image for printing
    image_path = create_task_html_image(task_data)
    if not image_path:
        # Fallback to PIL method
        image_path = create_task_image(task_data)

    if image_path:
        print(f"   Image saved: {image_path}")
        
        # Print if printer is available
        try:
            print("\n🖨️  Sending to printer...")
            print_to_thermal_printer(image_path)
            print("   ✅ Printed successfully!")
        except Exception as e:
            print(f"   ⚠️  Printing failed: {e}")
    else:
        print("   ❌ Could not create image")

    print("\n" + "=" * 50)
    print("Task generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()