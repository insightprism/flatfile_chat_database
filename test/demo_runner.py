#!/usr/bin/env python3
"""
Interactive demo runner that shows the CLI in action.
"""

import subprocess
import time
import sys

def run_interactive_demo():
    """Run the interactive demo with simulated input"""
    
    # Create input commands to simulate user interaction
    commands = [
        "alice",           # Username
        "Alice Johnson",   # Display name
        "n",              # New session
        "Python Learning Session",  # Session title
        "Hello! I'm learning Python.",  # First message
        "How do I create a list in Python?",  # Question
        "/search Python",  # Search command
        "/history 5",      # Show history
        "/stats",         # Show statistics
        "/sessions",      # List sessions
        "/help",          # Show help
        "/context",       # Show context
        "/export",        # Export session
        "/exit",          # Exit
    ]
    
    # Join commands with newlines for input
    input_data = "\n".join(commands) + "\n"
    
    print("=== RUNNING INTERACTIVE CHAT DEMO ===")
    print("Simulating user interaction...")
    print("-" * 50)
    
    # Run the interactive demo with piped input
    process = subprocess.Popen(
        ["python3", "interactive_chat_demo.py", "--theme", "colorful"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send all input at once
    stdout, stderr = process.communicate(input=input_data)
    
    # Print output
    print(stdout)
    
    if stderr:
        print("STDERR:", stderr)
    
    print("-" * 50)
    print("Demo completed!")
    
    # Show what was created
    print("\n📁 Files created during demo:")
    subprocess.run(["ls", "-la", "./chat_data/alice/"], check=False)

if __name__ == "__main__":
    run_interactive_demo()