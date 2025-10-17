"""
Daily journal management tool for Strands Agent.

This module provides functionality to create and manage daily journal entries with
rich text formatting, including task lists and notes. Journal entries are saved as
Markdown files in the cwd()/journal/ directory, organized by date.

Journal entries support both regular text notes and task management with checkboxes.
The tool provides a beautiful rich text interface with panels, tables, and formatting
to enhance the user experience when working with journal entries.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import journal

agent = Agent(tools=[journal])

# Write a new journal entry
agent.tool.journal(
    action="write",
    content="Today I worked on implementing the Strands SDK tools."
)

# Add a task to today's journal
agent.tool.journal(
    action="add_task",
    task="Complete the journal tool documentation"
)

# Read today's journal
result = agent.tool.journal(action="read")

# View a list of all journal entries
entries = agent.tool.journal(action="list")

# Read a specific date's journal
specific_entry = agent.tool.journal(
    action="read",
    date="2023-04-15"
)
```

See the journal function docstring for more details on available actions and parameters.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util

TOOL_SPEC = {
    "name": "journal",
    "description": "Create and manage daily journal entries with tasks and notes",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read", "list", "add_task"],
                    "description": "Action to perform (write/read/list/add_task)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write action)",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (defaults to today)",
                },
                "task": {
                    "type": "string",
                    "description": "Task to add (for add_task action)",
                },
            },
            "required": ["action"],
        }
    },
}


def ensure_journal_dir() -> Path:
    """
    Ensure journal directory exists.

    Creates the journal directory if it doesn't exist and returns
    the path to it.

    Returns:
        Path: The path to the journal directory
    """
    journal_dir = Path.cwd() / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    return journal_dir


def get_journal_path(date_str: Optional[str] = None) -> Path:
    """
    Get journal file path for given date.

    Args:
        date_str: Optional date string in YYYY-MM-DD format. If not provided,
                  current date is used.

    Returns:
        Path: Path to the journal file for the specified date
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return ensure_journal_dir() / f"{date_str}.md"


def create_rich_response(console: Console, action: str, result: Dict[str, Any]) -> None:
    """
    Create rich interface output for journal actions.

    This function generates visually appealing formatted output for different
    journal actions, using tables, panels, and styled text.

    Args:
        action: The journal action that was performed (write/read/list/add_task)
        result: Dictionary containing the action result data
    """
    if action == "write":
        panel = Panel(
            Text.assemble(
                ("✍️ Journal Entry Added\n\n", "bold magenta"),
                ("Time: ", "dim"),
                (datetime.now().strftime("%H:%M:%S"), "cyan"),
                ("\nDate: ", "dim"),
                (result["date"], "green"),
                ("\nPath: ", "dim"),
                (str(result["path"]), "blue"),
                ("\n\nContent:\n", "yellow"),
                (result["content"], "bright_white"),
            ),
            title="📔 Journal Update",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)

    elif action == "read":
        md = Markdown(result["content"])
        panel = Panel(
            md,
            title=f"📖 Journal Entry - {result['date']}",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)

    elif action == "list":
        table = Table(
            title="📚 Journal Entries",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            box=box.ROUNDED,
        )

        table.add_column("📅 Date", style="cyan", no_wrap=True)
        table.add_column("📝 Entries", style="green")
        table.add_column("✅ Tasks", style="yellow")

        for entry in result["entries"]:
            table.add_row(entry["date"], str(entry["entry_count"]), str(entry["task_count"]))

        console.print(table)

    elif action == "add_task":
        panel = Panel(
            Text.assemble(
                ("✅ Task Added\n\n", "bold green"),
                ("Time: ", "dim"),
                (datetime.now().strftime("%H:%M:%S"), "cyan"),
                ("\nDate: ", "dim"),
                (result["date"], "green"),
                ("\nTask: ", "dim"),
                (result["task"], "yellow"),
            ),
            title="📋 Task Management",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)


def journal(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Create and manage daily journal entries with tasks and notes.

    This tool allows you to write and read journal entries, add tasks, and list all
    available journal entries. Each journal is stored as a Markdown file in the
    cwd()/journal/ directory, organized by date.

    How It Works:
    ------------
    1. Journal entries are stored as Markdown files in the journal directory
    2. Each file is named with the date format YYYY-MM-DD.md
    3. Entries within a journal are timestamped with HH:MM:SS
    4. Tasks are stored with checkbox format (- [ ] task description)
    5. Rich formatting is applied when displaying journal content

    Journal Actions:
    --------------
    - write: Create a new entry in the specified date's journal
    - read: Display the content of a journal for a specific date
    - list: Show all available journal entries with stats
    - add_task: Add a task item to the specified date's journal

    Common Usage Scenarios:
    ---------------------
    - Daily note-taking and journaling
    - Task and todo list management
    - Progress tracking and reflection
    - Timestamped logging and record keeping

    Args:
        tool: The tool use object containing input parameters
            - action: Action to perform (write/read/list/add_task)
            - content: Content to write (required for write action)
            - date: Date in YYYY-MM-DD format (defaults to today)
            - task: Task to add (required for add_task action)

    Returns:
        ToolResult containing status and response content in the format:
        {
            "toolUseId": "<tool_use_id>",
            "status": "success|error",
            "content": [{"text": "Response message"}]
        }

        Success case: Returns confirmation of the action performed
        Error case: Returns information about what went wrong

    Notes:
        - If no date is specified, the current date is used
        - Each journal entry is automatically timestamped
        - The tool creates the journal directory if it doesn't exist
        - A rich text interface is provided for better user experience
        - Task completion status is maintained between sessions
    """
    console = console_util.create()

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    action = tool_input["action"]
    date = tool_input.get("date")

    try:
        if action == "write":
            content = tool_input.get("content")
            if not content:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "Content is required for write action"}],
                }

            journal_path = get_journal_path(date)
            timestamp = datetime.now().strftime("%H:%M:%S")

            with open(journal_path, "a") as f:
                f.write(f"\n## {timestamp}\n{content}\n")

            result = {
                "date": journal_path.stem,
                "path": str(journal_path),
                "content": content,
                "timestamp": timestamp,
            }

            create_rich_response(console, action, result)
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Added entry to journal: {journal_path}"}],
            }

        elif action == "read":
            journal_path = get_journal_path(date)
            if not journal_path.exists():
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"No journal found for date: {journal_path.stem}"}],
                }

            with open(journal_path) as f:
                content = f.read()

            result = {"date": journal_path.stem, "content": content}

            create_rich_response(console, action, result)
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": content}],
            }

        elif action == "list":
            journal_dir = ensure_journal_dir()
            journals = sorted(journal_dir.glob("*.md"))

            if not journals:
                return {
                    "toolUseId": tool_use_id,
                    "status": "success",
                    "content": [{"text": "No journal entries found"}],
                }

            entries = []
            for journal in journals:
                with open(journal) as f:
                    content = f.read()
                    entry_count = len([line for line in content.split("\n") if line.startswith("## ")])
                    task_count = content.count("- [ ]")
                    entries.append(
                        {
                            "date": journal.stem,
                            "entry_count": entry_count,
                            "task_count": task_count,
                        }
                    )

            result = {"entries": entries}
            create_rich_response(console, action, result)

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Listed {len(entries)} journal entries"}],
            }

        elif action == "add_task":
            task = tool_input.get("task")
            if not task:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "Task is required for add_task action"}],
                }

            journal_path = get_journal_path(date)
            timestamp = datetime.now().strftime("%H:%M:%S")

            with open(journal_path, "a") as f:
                f.write(f"\n## {timestamp} - Task\n- [ ] {task}\n")

            result = {"date": journal_path.stem, "task": task, "timestamp": timestamp}

            create_rich_response(console, action, result)
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Added task to journal: {journal_path}"}],
            }

        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Unknown action: {action}"}],
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
