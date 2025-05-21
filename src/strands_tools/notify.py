"""
Notification system for Strands Agent.

This module provides functionality to create and manage notifications from agents
and background tasks. Notifications can be displayed to the user, logged, and
optionally trigger system notifications.

Features:
- Send text notifications to the main agent's output
- Show system notifications (desktop/OS level)
- Customize notification priority and persistence
- Support for notification categories and filtering
- Integration with background tasks for completion alerts

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import notify, tasks

agent = Agent(tools=[notify, tasks])

# Send a simple notification
agent.tool.notify(
    message="Analysis complete!",
    title="Task Status"
)

# Send a high-priority notification with system alert
agent.tool.notify(
    message="Critical error detected in system monitoring",
    title="Alert",
    priority="high",
    show_system_notification=True
)

# Send notification from a background task
agent.tool.notify(
    message="Background research task found 5 relevant papers",
    title="Research Update",
    category="research",
    source="background_task_123"
)
```
"""

import json
import logging
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "notify",
    "description": "Send notifications to the main agent and optionally trigger system notifications",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The notification message content",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the notification",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "Priority level of the notification (default: normal)",
                },
                "category": {
                    "type": "string",
                    "description": "Category or type of notification for filtering/organization",
                },
                "show_system_notification": {
                    "type": "boolean",
                    "description": "Whether to also show a system notification (desktop/OS)",
                },
                "source": {
                    "type": "string",
                    "description": "Source of the notification (e.g., task ID, agent name)",
                },
                "persistent": {
                    "type": "boolean",
                    "description": "Whether this notification should be saved to history",
                },
            },
            "required": ["message"],
        }
    },
}

# Directory to store notification history
NOTIFICATIONS_DIR = Path.cwd() / "notifications"
NOTIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = NOTIFICATIONS_DIR / "notification_history.jsonl"


def _show_system_notification(title, message):
    """Display a system notification using the appropriate method for the OS."""
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            # Use osascript to show notification
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')

        elif system == "Linux":
            # Try to use notify-send if available
            os.system(f'notify-send "{title}" "{message}"')

        elif system == "Windows":
            # Use Windows toast notifications if available
            try:
                from win10toast import ToastNotifier

                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=5)
            except ImportError:
                # Fall back to a simpler approach using winsound
                import ctypes

                ctypes.windll.user32.MessageBoxW(0, message, title, 0)

        return True
    except Exception as e:
        logger.error(f"Failed to show system notification: {e}")
        return False


def _log_notification(notification_data):
    """Log notification to the history file."""
    if not notification_data.get("persistent", True):
        return

    # Add timestamp if not present
    if "timestamp" not in notification_data:
        notification_data["timestamp"] = datetime.now().isoformat()

    # Append to history file
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(notification_data) + "\n")


def _get_notification_history(limit=None, category=None, min_priority=None):
    """Get notification history with optional filtering."""
    if not os.path.exists(HISTORY_FILE):
        return []

    notifications = []
    priority_levels = {"low": 1, "normal": 2, "high": 3}
    min_priority_level = priority_levels.get(min_priority, 1) if min_priority else 1

    with open(HISTORY_FILE, "r") as f:
        for line in f:
            try:
                notification = json.loads(line.strip())

                # Apply filters
                if category and notification.get("category") != category:
                    continue

                if min_priority:
                    notification_priority = notification.get("priority", "normal")
                    if priority_levels.get(notification_priority, 2) < min_priority_level:
                        continue

                notifications.append(notification)
            except json.JSONDecodeError:
                continue

    # Sort by timestamp (newest first) and apply limit
    notifications.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if limit and limit > 0:
        notifications = notifications[:limit]

    return notifications


def notify(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Send notifications to the main agent and optionally trigger system notifications.

    This function creates notifications that can be displayed to the user, logged
    for future reference, and optionally trigger system-level notifications.

    Args:
        tool (ToolUse): Tool use object containing the following:
            - message: The notification message content
            - title: Optional title for the notification
            - priority: Priority level (low, normal, high)
            - category: Category of notification for filtering
            - show_system_notification: Whether to show system notification
            - source: Source of the notification
            - persistent: Whether to save this notification to history
            - action: Optional action - "send" (default) or "history" to get notification history
            - limit: Limit for history retrieval (used with action="history")
            - filter_category: Category filter for history (used with action="history")
            - min_priority: Minimum priority filter for history (used with action="history")
        **kwargs (Any): Additional keyword arguments

    Returns:
        ToolResult: Dictionary containing status and response content
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    # Check for history action
    action = tool_input.get("action", "send")

    if action == "history":
        # Return notification history
        limit = tool_input.get("limit", 10)
        category = tool_input.get("filter_category")
        min_priority = tool_input.get("min_priority")

        notifications = _get_notification_history(limit, category, min_priority)

        if not notifications:
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "No notifications found matching the criteria"}],
            }

        # Format notifications for display
        result_content = [{"text": f"Recent Notifications ({len(notifications)}):"}]

        for notification in notifications:
            priority_indicators = {"low": "📢", "normal": "🔔", "high": "⚠️"}
            priority = notification.get("priority", "normal")
            indicator = priority_indicators.get(priority, "🔔")

            timestamp = notification.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    formatted_time = timestamp
            else:
                formatted_time = "Unknown time"

            title = notification.get("title", "Notification")
            message = notification.get("message", "")
            source = notification.get("source", "")
            category = notification.get("category", "")

            notification_text = f"{indicator} [{formatted_time}] {title}: {message}"
            if source:
                notification_text += f" (from: {source})"
            if category:
                notification_text += f" [category: {category}]"

            result_content.append({"text": notification_text})

        return {"toolUseId": tool_use_id, "status": "success", "content": result_content}

    # Normal send notification action
    message = tool_input.get("message")
    if not message:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: message is required for notifications"}],
        }

    title = tool_input.get("title", "Notification")
    priority = tool_input.get("priority", "normal")
    category = tool_input.get("category", "general")
    show_system_notification = tool_input.get("show_system_notification", False)
    source = tool_input.get("source", "agent")
    persistent = tool_input.get("persistent", True)

    # Create notification data structure
    notification_data = {
        "message": message,
        "title": title,
        "priority": priority,
        "category": category,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "persistent": persistent,
    }

    # Log notification if persistent
    if persistent:
        _log_notification(notification_data)

    # Format notification message with priority indicator
    priority_indicators = {"low": "📢", "normal": "🔔", "high": "⚠️"}
    indicator = priority_indicators.get(priority, "🔔")

    formatted_message = f"{indicator} {title}: {message}"
    if source != "agent":
        formatted_message += f" (from: {source})"

    # Show system notification if requested
    system_notification_shown = False
    if show_system_notification:
        system_notification_shown = _show_system_notification(title, message)

    # Return notification content
    result_content = [{"text": formatted_message}]

    if show_system_notification:
        status_msg = "System notification displayed" if system_notification_shown else "System notification failed"
        result_content.append({"text": status_msg})

    return {"toolUseId": tool_use_id, "status": "success", "content": result_content}
