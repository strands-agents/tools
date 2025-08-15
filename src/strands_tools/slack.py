"""
Slack Integration Tool for Strands Agents
========================================

This module provides a comprehensive integration between Slack and Strands agents,
enabling AI-powered interactions within Slack workspaces through:

1. Real-time event processing via Socket Mode
2. Direct API access to all Slack methods
3. Simplified message sending with a dedicated tool

Key Features:
------------
- Socket Mode support for real-time events
- Access to all Slack API methods (auto-detected)
- Event history storage and retrieval
- Automatic message reaction handling
- Thread support for conversations
- Agent delegation for message processing
- Environment variable configuration
- Comprehensive error handling
- Dynamic toggling of auto-reply mode

Setup Requirements:
-----------------
1. Slack App with appropriate scopes:
   - chat:write
   - reactions:write
   - channels:history
   - app_mentions:read
   - channels:read
   - reactions:read
   - groups:read
   - im:read
   - mpim:read

2. Environment variables:
   - SLACK_BOT_TOKEN: xoxb-... token from Slack app
   - SLACK_APP_TOKEN: xapp-... token with Socket Mode enabled
   - STRANDS_SLACK_LISTEN_ONLY_TAG (optional): Only process messages with this tag
   - STRANDS_SLACK_AUTO_REPLY (optional): Set to "true" to enable automatic replies

Usage Examples:
-------------
# Basic setup with Strands agent
```python
from strands import Agent
from strands_tools import slack

# Create agent with Slack tool
agent = Agent(tools=[slack])

# Use the agent to interact with Slack
result = agent.tool.slack(
    action="chat_postMessage",
    parameters={"channel": "C123456", "text": "Hello from Strands!"}
)

# For simple message sending, use the dedicated tool
result = agent.tool.slack_send_message(
    channel="C123456",
    text="Hello from Strands!",
    thread_ts="1234567890.123456"  # Optional - reply in thread
)

# Start Socket Mode to listen for real-time events
agent.tool.slack(action="start_socket_mode")

# Get recent events from Slack
events = agent.tool.slack(
    action="get_recent_events",
    parameters={"count": 10}
)

# Toggle auto-reply mode using the environment tool
agent.tool.environment(
    action="set",
    name="STRANDS_SLACK_AUTO_REPLY",
    value="true"  # Set to "false" to disable auto-replies
)
```

Socket Mode:
----------
The tool includes a socket mode handler that connects to Slack's real-time
messaging API and processes events through a Strands agent. When enabled, it:

1. Listens for incoming Slack events
2. Adds a "thinking" reaction to show processing
3. Uses a Strands agent to generate responses
4. Removes the "thinking" reaction and adds a completion reaction
5. Stores events for later retrieval

Real-time events are stored in a local file system at: ./slack_events/events.jsonl

Auto-Reply Mode:
--------------
You can control whether the agent automatically sends replies to Slack or simply
processes messages without responding:

- Set STRANDS_SLACK_AUTO_REPLY=true: Agent will automatically send responses to Slack
- Default behavior (false): Agent will process messages but won't automatically reply

This feature allows you to:
1. Run in "listen-only" mode to monitor without responding
2. Toggle auto-reply behavior dynamically using the environment tool
3. Implement custom reply logic using the slack_send_message tool

Error Handling:
------------
The tool includes comprehensive error handling for:
- API rate limiting
- Network issues
- Authentication problems
- Malformed requests
- Socket disconnections

When errors occur, appropriate error messages are returned and logged.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from slack_bolt import App
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.client import WebClient
from strands import Agent, tool

# Configure logging
logger = logging.getLogger(__name__)

# System prompt for Slack communications
SLACK_SYSTEM_PROMPT = """
You are an AI assistant integrated with a Slack workspace. Important guidelines:

1. MESSAGE IDENTIFICATION:
   - You'll receive ALL events including your own messages echoed back
   - NEVER respond to your own messages - check bot_id/user_id carefully
   - Your messages typically have a bot_id or app_user_id field
   - The user field in your messages may be different from regular user messages

2. INTERACTION CAPABILITIES:
   - Send messages with slack_send_message(channel, text, thread_ts)
   - Add reactions with slack(action="reactions_add", parameters={"channel": "...", "name": "emoji_name", "timestamp": "..."})
   - Upload images with slack(action="files_upload_v2", parameters={"channel_id": "...", "file": "...", "title": "..."})
   - Upload audio with slack(action="files_upload_v2", parameters={"channel_id": "...", "file": "/path/to/audio.mp3", "title": "Audio Title", "filetype": "mp3"})
   - Create rich messages with blocks using chat_postMessage action

3. CONVERSATION FLOW:
   - Maintain context across messages in threads
   - Use thread_ts parameter to keep conversations organized
   - Prefer thread replies for multi-turn conversations

4. CONTENT GUIDELINES:
   - Use Slack markdown for formatting (*bold*, _italic_, `code`)
   - Keep responses concise and formatted for chat readability
   - Use emoji reactions as appropriate signals
   - Follow communication priority: Speaking/Audio > Text speaking > Images > Plain text > Reactions
   - When possible, prioritize sending audio messages over other forms of communication

Use slack send message tool to communicate back.
"""  # noqa: E501

# Initialize Slack app and client based on environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

# Initialize clients as None first
app = None
client = None
socket_client = None

# Event storage configuration
EVENTS_DIR = Path.cwd() / "slack_events"
EVENTS_FILE = EVENTS_DIR / "events.jsonl"

# Make sure events directory exists
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def initialize_slack_clients():
    """
    Initialize Slack clients if tokens are available.

    This function sets up three global clients:
    1. app: Slack Bolt application for handling events
    2. client: WebClient for making Slack API calls
    3. socket_client: SocketModeClient for real-time events

    Environment Variables:
        SLACK_BOT_TOKEN: The bot token starting with 'xoxb-'
        SLACK_APP_TOKEN: The app-level token starting with 'xapp-'

    Returns:
        tuple: (success, error_message)
            - success (bool): True if initialization was successful
            - error_message (str): None if successful, error details otherwise

    Example:
        success, error = initialize_slack_clients()
        if not success:
            print(f"Failed to initialize Slack: {error}")
    """
    global app, client, socket_client

    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        return (
            False,
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables",
        )

    try:
        app = App(token=SLACK_BOT_TOKEN)
        client = WebClient(token=SLACK_BOT_TOKEN)
        socket_client = SocketModeClient(app_token=SLACK_APP_TOKEN, web_client=client)
        return True, None
    except Exception as e:
        return False, f"Error initializing Slack clients: {str(e)}"


class SocketModeHandler:
    """
    Handle Socket Mode connections and events for real-time Slack interactions.

    This class manages the connection to Slack's Socket Mode API, which allows
    for real-time event processing without requiring a public-facing endpoint.

    Key Features:
    - Automatic connection management
    - Event processing with Strands agents
    - Event storage for historical access
    - Reaction-based status indicators (thinking, completed, error)
    - Thread-based conversation support
    - Error handling with visual feedback

    Typical Usage:
    ```python
    # Initialize the handler
    handler = SocketModeHandler()

    # Start listening for events
    handler.start()

    # Process events for a while...

    # Stop the connection when done
    handler.stop()
    ```

    Events Processing Flow:
    1. Event received from Slack
    2. Event acknowledged immediately
    3. Event stored to local filesystem
    4. "thinking_face" reaction added to show processing
    5. Event processed by Strands agent
    6. "thinking_face" reaction removed
    7. "white_check_mark" reaction added on success
    8. Error handling with "x" reaction if needed
    """

    def __init__(self):
        self.client = None
        self.is_connected = False
        self.agent = None

    def _setup_client(self):
        """Set up the socket client if not already initialized."""
        if socket_client is None:
            success, error_message = initialize_slack_clients()
            if not success:
                raise ValueError(error_message)
        self.client = socket_client
        self._setup_listeners()

    def _setup_listeners(self):
        """Set up event listeners for Socket Mode."""

        def process_event(client: SocketModeClient, req: SocketModeRequest):
            """Process incoming Socket Mode events."""
            logger.info("🎯 Socket Mode Event Received!")
            logger.info(f"Event Type: {req.type}")

            # Always acknowledge the request first
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            logger.info("✅ Event Acknowledged")

            try:
                # Store event in file system
                event_data = {
                    "event_type": req.type,
                    "payload": req.payload,
                    "timestamp": time.time(),
                    "envelope_id": req.envelope_id,
                }

                # Save event to disk
                EVENTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(EVENTS_FILE, "a") as f:
                    f.write(json.dumps(event_data) + "\n")

                # Process the event based on type
                event = req.payload.get("event", {})

                # Handle message events
                if req.type == "events_api" and event.get("type") == "message" and not event.get("subtype"):
                    logger.info("💬 Processing Message Event")
                    self._process_message(event)

                # Handle interactive events
                elif req.type == "interactive":
                    logger.info("🔄 Processing Interactive Event")
                    interactive_context = {
                        "type": "interactive",
                        "channel": req.payload.get("channel", {}).get("id"),
                        "user": req.payload.get("user", {}).get("id"),
                        "ts": req.payload.get("message", {}).get("ts"),
                        "actions": req.payload.get("actions", []),
                        "full_payload": req.payload,
                    }
                    self._process_interactive(interactive_context)

                logger.info("✅ Event Processing Complete")

            except Exception as e:
                logger.error(f"Error processing socket mode event: {e}", exc_info=True)

        # Add the event listener
        self.client.socket_mode_request_listeners.append(process_event)

    def _process_message(self, event):
        """Process a message event using a Strands agent."""
        # Get bot info once and cache it
        if not hasattr(self, "bot_info"):
            try:
                self.bot_info = client.auth_test()
            except Exception as e:
                logger.error(f"Error getting bot info: {e}")
                self.bot_info = {"user_id": None, "bot_id": None}

        # Skip processing if this is our own message
        if event.get("bot_id") or event.get("user") == self.bot_info.get("user_id") or "app_id" in event:
            logger.info("Skipping own message")
            return

        tools = list(self.agent.tool_registry.registry.values())
        trace_attributes = self.agent.trace_attributes

        agent = Agent(
            model=self.agent.model,
            messages=[],
            system_prompt=f"{self.agent.system_prompt}\n{SLACK_SYSTEM_PROMPT}",
            tools=tools,
            callback_handler=self.agent.callback_handler,
            trace_attributes=trace_attributes,
        )

        channel_id = event.get("channel")
        text = event.get("text", "")
        user = event.get("user")
        ts = event.get("ts")

        # Add thinking reaction
        try:
            if client:
                client.reactions_add(name="thinking_face", channel=channel_id, timestamp=ts)
        except Exception as e:
            logger.error(f"Error adding thinking reaction: {e}")

        # Get recent events for context
        slack_default_event_count = int(os.getenv("SLACK_DEFAULT_EVENT_COUNT", "42"))
        recent_events = self._get_recent_events(slack_default_event_count)
        event_context = f"\nRecent Slack Events: {json.dumps(recent_events)}" if recent_events else ""

        # Process with agent
        try:
            # Check if we should process this message (based on environment tag)
            listen_only_tag = os.environ.get("STRANDS_SLACK_LISTEN_ONLY_TAG")
            if listen_only_tag and listen_only_tag not in text:
                logger.info(f"Skipping message - does not contain tag: {listen_only_tag}")
                return

            # Refresh the system prompt with latest context handled from Slack events
            agent.system_prompt = (
                f"{SLACK_SYSTEM_PROMPT}\n\nEvent Context:\nCurrent: {json.dumps(event)}{event_context}"
            )

            # Process with agent
            response = agent(f"[Channel: {channel_id}] User {user} says: {text}")

            # If we have a valid response, send it back to Slack
            if response and str(response).strip():
                if client:
                    # Check if auto-reply is enabled
                    if os.getenv("STRANDS_SLACK_AUTO_REPLY", "false").lower() == "true":
                        client.chat_postMessage(
                            channel=channel_id,
                            text=str(response).strip(),
                            thread_ts=ts,
                        )

                    # Remove thinking reaction
                    client.reactions_remove(name="thinking_face", channel=channel_id, timestamp=ts)

                    # Add completion reaction
                    client.reactions_add(name="white_check_mark", channel=channel_id, timestamp=ts)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

            # Try to send error message to channel
            if client:
                try:
                    # Remove thinking reaction
                    client.reactions_remove(name="thinking_face", channel=channel_id, timestamp=ts)

                    # Add error reaction and message
                    client.reactions_add(name="x", channel=channel_id, timestamp=ts)

                    # Only send error message if auto-reply is enabled
                    if os.getenv("STRANDS_SLACK_AUTO_REPLY", "false").lower() == "true":
                        client.chat_postMessage(
                            channel=channel_id,
                            text=f"Error processing message: {str(e)}",
                            thread_ts=ts,
                        )
                except Exception as e2:
                    logger.error(f"Error sending error message: {e2}")

    def _process_interactive(self, event):
        """Process an interactive event."""
        # Process interactive events similar to messages
        if client and self.agent:
            tools = list(self.agent.tool_registry.registry.values())

            agent = Agent(
                model=self.agent.model,
                messages=[],
                system_prompt=SLACK_SYSTEM_PROMPT,
                tools=tools,
                callback_handler=self.agent.callback_handler,
            )

            channel_id = event.get("channel")
            actions = event.get("actions", [])
            ts = event.get("ts")

            # Create context message for the agent
            interaction_text = f"Interactive event from user {event.get('user')}. Actions: {actions}"

            try:
                agent.system_prompt = f"{SLACK_SYSTEM_PROMPT}\n\nInteractive Context:\n{json.dumps(event, indent=2)}"
                response = agent(interaction_text)

                # Only send a response if auto-reply is enabled
                if os.getenv("STRANDS_SLACK_AUTO_REPLY", "false").lower() == "true":
                    client.chat_postMessage(
                        channel=channel_id,
                        text=str(response).strip(),
                        thread_ts=ts,
                    )

                # Add a reaction to indicate completion
                client.reactions_add(name="white_check_mark", channel=channel_id, timestamp=ts)

            except Exception as e:
                logger.error(f"Error processing interactive event: {e}", exc_info=True)
                try:
                    # Add error reaction
                    client.reactions_add(name="x", channel=channel_id, timestamp=ts)
                except Exception as e2:
                    logger.error(f"Error adding error reaction: {e2}")

    def _get_recent_events(self, count: int) -> List[Dict[str, Any]]:
        """Get recent events from the file system."""
        if not EVENTS_FILE.exists():
            return []

        try:
            with open(EVENTS_FILE, "r") as f:
                # Get the last 'count' events
                lines = f.readlines()[-count:]
                events = []
                for line in lines:
                    try:
                        event_data = json.loads(line.strip())
                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue
                return events
        except Exception as e:
            logger.error(f"Error reading events file: {e}")
            return []

    def start(self, agent):
        """Start the Socket Mode connection."""
        logger.info("🚀 Starting Socket Mode Connection...")

        self.agent = agent

        if not self.is_connected:
            try:
                self._setup_client()
                self.client.connect()
                self.is_connected = True
                logger.info("✅ Socket Mode connection established!")
                return True
            except Exception as e:
                logger.error(f"❌ Error starting Socket Mode: {str(e)}")
                return False
        logger.info("ℹ️ Already connected, no action needed")
        return True

    def stop(self):
        """Stop the Socket Mode connection."""
        if self.is_connected and self.client:
            try:
                self.client.close()
                self.is_connected = False
                logger.info("Socket Mode connection closed")
                return True
            except Exception as e:
                logger.error(f"Error stopping Socket Mode: {e}", exc_info=True)
                return False
        return True


# Initialize socket handler
socket_handler = SocketModeHandler()


@tool
def slack(action: str, parameters: Dict[str, Any] = None, agent=None) -> str:
    """Slack integration for messaging, events, and interactions.

    This tool provides complete access to Slack's API methods and real-time
    event handling through a unified interface. It enables Strands agents to
    communicate with Slack workspaces, respond to messages, add reactions,
    manage channels, and more.

    Action Categories:
    -----------------
    1. Slack API Methods: Any method from the Slack Web API (e.g., chat_postMessage)
       Direct passthrough to Slack's API using the parameters dictionary

    2. Socket Mode Actions:
       - start_socket_mode: Begin listening for real-time events
       - stop_socket_mode: Stop the Socket Mode connection

    3. Event Management:
       - get_recent_events: Retrieve stored events from history

    Args:
        action: The action to perform. Can be:
            - Any valid Slack API method (chat_postMessage, reactions_add, etc.)
            - "start_socket_mode": Start listening for real-time events
            - "stop_socket_mode": Stop listening for real-time events
            - "get_recent_events": Retrieve recent events from storage
        parameters: Parameters for the action. For Slack API methods, these are
                  passed directly to the API. For custom actions, specific
                  parameters may be needed.

    Returns:
        str: Result of the requested action, typically containing a success/error
             status and relevant details or response data.

    Examples:
    --------
    # Send a message
    result = slack(
        action="chat_postMessage",
        parameters={{
            "channel": "C0123456789",
            "text": "Hello from Strands!",
            "blocks": [{{"type": "section", "text": {{"type": "mrkdwn", "text": "*Bold* message"}}}}]
        }}
    )

    # Add a reaction to a message
    result = slack(
        action="reactions_add",
        parameters={{
            "channel": "C0123456789",
            "timestamp": "1234567890.123456",
            "name": "thumbsup"
        }}
    )

    # Start listening for real-time events
    result = slack(action="start_socket_mode")

    # Get recent events
    result = slack(action="get_recent_events", parameters={{"count": 10}})

    Notes:
    -----
    - Slack event stream include your own messages, do not reply yourself.
    - Required environment variables: SLACK_BOT_TOKEN, SLACK_APP_TOKEN
    - Optional environment variables:
      - STRANDS_SLACK_AUTO_REPLY: Set to "true" to enable automatic replies to messages
      - STRANDS_SLACK_LISTEN_ONLY_TAG: Only process messages containing this tag
      - SLACK_DEFAULT_EVENT_COUNT: Number of events to retrieve by default (default: 42)
    - Events are stored locally at ./slack_events/events.jsonl
    - See Slack API documentation for all available methods and parameters
    """
    # Initialize Slack clients if needed
    if action != "get_recent_events" and client is None:
        success, error_message = initialize_slack_clients()
        if not success:
            return f"Error: {error_message}"

    # Set default parameters
    if parameters is None:
        parameters = {}

    try:
        # Handle Socket Mode actions
        if action == "start_socket_mode":
            if socket_handler.start(agent):
                return "✅ Socket Mode connection established and ready to receive real-time events"
            return "❌ Failed to establish Socket Mode connection"

        elif action == "stop_socket_mode":
            if socket_handler.stop():
                return "✅ Socket Mode connection closed"
            return "❌ Failed to close Socket Mode connection"

        # Handle event retrieval
        elif action == "get_recent_events":
            count = parameters.get("count", 5)
            if not EVENTS_FILE.exists():
                return "No events found in storage"

            with open(EVENTS_FILE, "r") as f:
                lines = f.readlines()[-count:]
                events = []
                for line in lines:
                    try:
                        event_data = json.loads(line.strip())
                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue

                # Always return a string, never None
                if events:
                    return f"Slack events: {json.dumps(events)}"
                else:
                    return "No valid events found in storage"

        # Standard Slack API methods
        else:
            # Check if method exists in the Slack client
            if hasattr(client, action) and callable(getattr(client, action)):
                method = getattr(client, action)
                response = method(**parameters)
                return f"✅ {action} executed successfully\n{json.dumps(response.data, indent=2)}"
            else:
                return f"❌ Unknown Slack action: {action}"

    except SlackApiError as e:
        logger.error(f"Slack API Error in {action}: {e.response['error']}")
        return f"Error: {e.response['error']}\nError code: {e.response.get('error')}"
    except Exception as e:
        logger.error(f"Error executing {action}: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def slack_send_message(channel: str, text: str, thread_ts: str = None) -> str:
    """Send a message to a Slack channel.

    This is a simplified interface for the most common Slack operation: sending messages.
    It wraps the Slack API's chat_postMessage method with a more direct interface,
    making it easier to send basic messages to channels or threads.

    Args:
        channel: The channel ID to send the message to. This should be the Slack
                channel ID (e.g., "C0123456789") rather than the channel name.
                To get a list of available channels and their IDs, use:
                slack(action="conversations_list")

        text: The message text to send. This can include Slack markdown formatting
              such as *bold*, _italics_, ~strikethrough~, `code`, and ```code blocks```,
              as well as @mentions and channel links.

        thread_ts: Optional thread timestamp to reply in a thread. When provided,
                  the message will be sent as a reply to the specified thread
                  rather than as a new message in the channel.

    Returns:
        str: Result message indicating success or failure, including the timestamp
             of the sent message on success.

    Examples:
    --------
    # Send a simple message to a channel
    result = slack_send_message(
        channel="C0123456789",
        text="Hello from Strands!"
    )

    # Reply to a thread
    result = slack_send_message(
        channel="C0123456789",
        text="This is a thread reply",
        thread_ts="1234567890.123456"
    )

    # Send a message with formatting
    result = slack_send_message(
        channel="C0123456789",
        text="*Important*: Please review this _document_."
    )

    Notes:
    -----
    - For more advanced message formatting using blocks, attachments, or other
      Slack features, use the main slack tool with the chat_postMessage action.
    - This function automatically ensures the Slack clients are initialized.
    - Channel IDs typically start with 'C', direct message IDs with 'D'.
    """
    if client is None:
        success, error_message = initialize_slack_clients()
        if not success:
            return f"Error: {error_message}"

    try:
        params = {"channel": channel, "text": text}
        if thread_ts:
            params["thread_ts"] = thread_ts

        response = client.chat_postMessage(**params)
        if response and response.get("ts"):
            return f"Message sent successfully. Timestamp: {response['ts']}"
        else:
            return "Message sent but no timestamp received from Slack API"
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        return f"Error sending message: {error_msg}"
