"""
User handoff tool for Strands Agent.

This module provides functionality to hand off control from the agent to the user,
allowing for human intervention in automated workflows. It's particularly useful for:

1. Getting user confirmation before proceeding with critical actions
2. Requesting additional information that the agent cannot determine
3. Allowing users to review and approve agent decisions
4. Creating interactive workflows where human input is required
5. Debugging and troubleshooting by pausing execution for user review

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import handoff_to_user

agent = Agent(tools=[handoff_to_user])

# Request user input and continue
response = agent.tool.handoff_to_user(
    message="I need your approval to proceed with deleting these files. Type 'yes' to confirm.",
    breakout_of_loop=False
)

# Stop execution and hand off to user
agent.tool.handoff_to_user(
    message="Task completed. Please review the results and take any necessary follow-up actions.",
    breakout_of_loop=True
)
```

The handoff tool can either pause for user input or completely stop the event loop,
depending on the breakout_of_loop parameter.
"""

import logging
from typing import Any

from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

# Initialize logging and console
logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "handoff_to_user",
    "description": "Hand off control from agent to user for confirmation, input, or complete task handoff",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to display to the user with context and instructions",
                },
                "breakout_of_loop": {
                    "type": "boolean",
                    "description": "Whether to stop the event loop (True) or wait for user input (False)",
                    "default": False,
                },
            },
            "required": ["message"],
        }
    },
}


def handoff_to_user(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Hand off control from the agent to the user for human intervention.

    This tool allows the agent to pause execution and request human input or approval.
    It can either wait for user input and continue, or completely stop the event loop
    to hand off control to the user.

    How It Works:
    ------------
    1. Displays a clear indication that the agent is requesting user handoff
    2. Shows the agent's message to the user (should include context and instructions)
    3. If breakout_of_loop is True: Sets the stop_event_loop flag to terminate gracefully
    4. If breakout_of_loop is False: Waits for user input and returns the response

    Common Usage Scenarios:
    ---------------------
    - User confirmation: Get approval before executing critical operations
    - Information gathering: Request additional details the agent cannot determine
    - Decision points: Allow users to choose between multiple options
    - Review and approval: Pause for user to review agent's work
    - Interactive workflows: Create human-in-the-loop processes
    - Debugging: Stop execution for troubleshooting and manual intervention

    Args:
        tool: The tool use object containing the tool input parameters
            - message: The message to display to the user. Should include:
                * Context about what the agent was doing
                * What the agent needs from the user
                * Clear instructions on how to respond
                * Any relevant details for decision making
            - breakout_of_loop: Whether to stop the event loop after displaying the message.
                * True: Stop the event loop completely (agent hands off control)
                * False: Wait for user input and continue with the response (default)
        **kwargs: Additional keyword arguments
            - request_state: Dictionary containing the current request state

    Returns:
        ToolResult containing:
            - toolUseId: The unique identifier for this tool use request
            - status: "success" or "error"
            - content: List with result text
                * If breakout_of_loop=True: Confirmation that handoff was initiated
                * If breakout_of_loop=False: The user's input response

    Examples:
        # Request user confirmation
        handoff_to_user({
            "toolUseId": "123",
            "input": {
                "message": "I'm about to delete 5 files. Type 'confirm' to proceed or 'cancel' to stop.",
                "breakout_of_loop": False
            }
        })

        # Complete handoff to user
        handoff_to_user({
            "toolUseId": "456",
            "input": {
                "message": "Analysis complete. Results saved to report.pdf. Please review and distribute as needed.",
                "breakout_of_loop": True
            }
        })

    Notes:
        - Always provide clear, actionable messages to users
        - Use breakout_of_loop=True for final handoffs or when agent work is complete
        - Use breakout_of_loop=False for mid-workflow user input
        - The tool handles the technical details of event loop control
        - This tool only affects the current event loop cycle, not the entire application
        - The handoff is graceful, allowing current operations to complete
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    request_state = kwargs.get("request_state", {})

    # Extract parameters
    message = tool_input.get("message", "Agent requesting user handoff")
    breakout_of_loop = tool_input.get("breakout_of_loop", False)

    # Display handoff notification using rich console
    console = console_util.create()
    console.print()
    handoff_panel = Panel(
        f"🤝 [bold green]AGENT REQUESTING USER HANDOFF[/bold green]\n\n{message}", border_style="green", padding=(1, 2)
    )
    console.print(handoff_panel)

    if breakout_of_loop:
        # Stop the event loop and hand off control
        request_state["stop_event_loop"] = True

        stop_panel = Panel(
            "🛑 [bold red]Agent execution stopped. Control handed off to user.[/bold red]",
            border_style="red",
            padding=(0, 2),
        )
        console.print(stop_panel)
        console.print()

        logger.info(f"Agent handoff initiated with message: {message}")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Agent handoff completed. Message displayed to user: {message}"}],
        }
    else:
        # Wait for user input and continue
        try:
            user_response = get_user_input(
                f"<bold>Agent requested user input:</bold> {message}\n<bold>Your response:</bold> "
            ).strip()

            console.print()

            logger.info(f"User handoff completed. User response: {user_response}")

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"User response received: {user_response}"}],
            }
        except KeyboardInterrupt:
            console.print()
            interrupt_panel = Panel(
                "🛑 [bold red]User interrupted. Stopping execution.[/bold red]", border_style="red", padding=(0, 2)
            )
            console.print(interrupt_panel)
            console.print()
            request_state["stop_event_loop"] = True

            logger.info("User interrupted handoff. Execution stopped.")

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "User interrupted handoff. Execution stopped."}],
            }
        except Exception as e:
            logger.error(f"Error during user handoff: {e}")

            error_panel = Panel(
                f"❌ [bold red]Error getting user input: {e}[/bold red]", border_style="red", padding=(0, 2)
            )
            console.print(error_panel)
            console.print()

            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error during user handoff: {str(e)}"}],
            }
