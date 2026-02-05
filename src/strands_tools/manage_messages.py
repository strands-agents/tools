"""Conversation history management for Strands Agent.

This module provides direct manipulation of agent.messages with turn-aware operations.
It works directly with the SDK's message list rather than reimplementing message handling.

A "turn" is a complete interaction cycle:
1. User message (input)
2. Assistant message (may contain toolUse)
3. User message with toolResult (if tool was used)
4. Assistant message (final response)

Key Features:
- list: View conversation by turns or filtered by role
- list_tools: View all tool calls with IDs for granular management
- stats: Get conversation statistics
- export/import: Persist and restore conversations
- drop: Remove complete turns safely
- drop_tools: Remove specific tool calls by ID or name
- compact: Strip toolUse/toolResult from turns (keep text content)
- clear: Reset conversation history

Usage Examples:
```python
from strands import Agent
from strands_tools import manage_messages

agent = Agent(tools=[manage_messages])

# View conversation
agent.tool.manage_messages(action="list")
agent.tool.manage_messages(action="list", role="user")

# View tool calls
agent.tool.manage_messages(action="list_tools")

# Get statistics
agent.tool.manage_messages(action="stats")

# Export/import for persistence
agent.tool.manage_messages(action="export", path="/tmp/conversation.json")
agent.tool.manage_messages(action="import", path="/tmp/conversation.json")

# Drop turns to manage context window
agent.tool.manage_messages(action="drop", turns="0,2,5")
agent.tool.manage_messages(action="drop", start=0, end=5)

# Drop specific tool calls
agent.tool.manage_messages(action="drop_tools", tool_ids="tooluse_abc123,tooluse_def456")
agent.tool.manage_messages(action="drop_tools", tool_name="shell")  # drop all shell calls

# Compact turns (strip tool blocks, keep text - reduces context size)
agent.tool.manage_messages(action="compact")  # auto-compact all but last 3 turns
agent.tool.manage_messages(action="compact", turns="0,1,2")  # specific turns

# Clear all messages
agent.tool.manage_messages(action="clear")
```
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from strands import tool

logger = logging.getLogger(__name__)

# Default summary length - configurable via STRANDS_MESSAGE_SUMMARY_LEN env var
DEFAULT_SUMMARY_LEN = int(os.getenv("STRANDS_MESSAGE_SUMMARY_LEN", "80"))


def _parse_turns(messages: List[Dict]) -> List[Tuple[int, int]]:
    """Parse messages into turns (start_idx, end_idx).

    A turn is a complete user interaction cycle:
    - Starts with a user message (NOT a toolResult)
    - Includes ALL subsequent tool cycles (assistant→toolResult→assistant→...)
    - Ends when we hit another user message (non-toolResult) or end of messages

    This correctly handles multi-tool chains where one user query triggers
    multiple sequential or parallel tool calls.
    """
    turns = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Turn starts with a user message that is NOT a toolResult
        if msg["role"] == "user":
            is_tool_result = any("toolResult" in b for b in msg.get("content", []))

            if is_tool_result:
                # This is a toolResult - skip, it's part of a turn we're building
                i += 1
                continue

            # Start of a new turn (fresh user query)
            start = i
            i += 1

            # Consume all tool cycles until we hit a new user query or end
            while i < len(messages):
                if messages[i]["role"] == "assistant":
                    i += 1  # consume assistant response

                    # Check if there's a toolResult following
                    if i < len(messages) and messages[i]["role"] == "user":
                        has_tool_result = any("toolResult" in b for b in messages[i].get("content", []))
                        if has_tool_result:
                            i += 1  # consume toolResult, continue loop for more cycles
                            continue
                        else:
                            # New user query - end this turn
                            break
                    else:
                        # End of messages or unexpected - end turn
                        break
                else:
                    # Not assistant - shouldn't happen but handle gracefully
                    break

            turns.append((start, i))
        else:
            # Skip orphaned assistant messages (shouldn't happen)
            i += 1

    return turns

def _summarize(content: list, max_len: Optional[int] = None) -> str:
    """Summarize content blocks.
    
    Args:
        content: List of content blocks to summarize
        max_len: Maximum length for text preview. Defaults to STRANDS_MESSAGE_SUMMARY_LEN env var or 80.
    """
    if max_len is None:
        max_len = DEFAULT_SUMMARY_LEN
    parts = []
    for block in content[:3]:
        if "text" in block:
            text = block["text"][:max_len]
            parts.append(f'"{text}{"..." if len(block["text"]) > max_len else ""}"')
        elif "toolUse" in block:
            parts.append(f"toolUse:{block['toolUse']['name']}")
        elif "toolResult" in block:
            parts.append(f"toolResult:{block['toolResult']['toolUseId'][:8]}...")
        elif "image" in block:
            parts.append("[image]")
        elif "reasoningContent" in block:
            parts.append("[reasoning]")

    if len(content) > 3:
        parts.append(f"+{len(content) - 3} more")

    return " | ".join(parts) if parts else "(empty)"


def _get_all_tool_calls(messages: List[Dict]) -> List[Dict]:
    """Extract all tool calls with their metadata.

    Returns list of dicts with:
    - tool_use_id: The unique ID
    - tool_name: Name of the tool
    - msg_idx: Message index containing toolUse
    - has_result: Whether toolResult exists
    - result_msg_idx: Message index containing toolResult (if exists)
    - args_preview: Preview of arguments
    - result_preview: Preview of result (if exists)
    """
    tool_calls = []
    tool_results = {}  # Map tool_use_id -> (msg_idx, result_preview)

    # First pass: collect all toolResults
    for msg_idx, msg in enumerate(messages):
        for block in msg.get("content", []):
            if "toolResult" in block:
                tr = block["toolResult"]
                result_text = ""
                for c in tr.get("content", []):
                    if "text" in c:
                        result_text = c["text"][:100]
                        break
                tool_results[tr["toolUseId"]] = (msg_idx, result_text, tr.get("status", "unknown"))

    # Second pass: collect all toolUse and match with results
    for msg_idx, msg in enumerate(messages):
        for block in msg.get("content", []):
            if "toolUse" in block:
                tu = block["toolUse"]
                tool_use_id = tu["toolUseId"]

                # Get args preview
                args = tu.get("input", {})
                if isinstance(args, dict):
                    args_preview = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:3])
                else:
                    args_preview = str(args)[:100]

                result_info = tool_results.get(tool_use_id)

                tool_calls.append(
                    {
                        "tool_use_id": tool_use_id,
                        "tool_name": tu["name"],
                        "msg_idx": msg_idx,
                        "has_result": result_info is not None,
                        "result_msg_idx": result_info[0] if result_info else None,
                        "result_status": result_info[2] if result_info else None,
                        "args_preview": args_preview,
                        "result_preview": result_info[1] if result_info else None,
                    }
                )

    return tool_calls


def _get_pending_tool_use_ids(messages: List[Dict]) -> List[str]:
    """Find all toolUse IDs that don't have corresponding toolResults.

    Scans through messages to find any toolUse blocks whose IDs are not
    matched by a subsequent toolResult block.

    Returns:
        List of orphaned toolUse IDs
    """
    tool_use_ids = set()
    tool_result_ids = set()

    for msg in messages:
        for block in msg.get("content", []):
            if "toolUse" in block:
                tool_use_ids.add(block["toolUse"]["toolUseId"])
            elif "toolResult" in block:
                tool_result_ids.add(block["toolResult"]["toolUseId"])

    # Return IDs that have toolUse but no toolResult
    return list(tool_use_ids - tool_result_ids)


def _remove_tool_blocks(messages: List[Dict], tool_ids_to_remove: Set[str]) -> List[Dict]:
    """Remove specific toolUse and toolResult blocks from messages.

    Removes both the toolUse block from assistant messages and the
    corresponding toolResult block from user messages.

    IMPORTANT: Messages containing 'thinking' or 'redacted_thinking' blocks
    are kept unchanged - Bedrock/Claude requires these blocks to remain unmodified.

    Handles cleanup of empty messages after removal.
    """
    result = []

    for msg in messages:
        # Check if message contains thinking blocks - these MUST remain unchanged
        has_thinking = any("thinking" in block or "redacted_thinking" in block for block in msg.get("content", []))

        if has_thinking:
            # Keep message entirely unchanged to preserve thinking blocks
            result.append(msg)
            continue

        new_content = []

        for block in msg.get("content", []):
            keep = True

            if "toolUse" in block:
                if block["toolUse"]["toolUseId"] in tool_ids_to_remove:
                    keep = False
            elif "toolResult" in block:
                if block["toolResult"]["toolUseId"] in tool_ids_to_remove:
                    keep = False

            if keep:
                new_content.append(block)

        # Only keep message if it has content
        if new_content:
            result.append({**msg, "content": new_content})
        # If message is now empty and was a toolResult-only message, skip it
        # But if it was a user query or assistant response with text, we'd keep it

    return result


def _strip_tool_blocks_from_turns(
    messages: List[Dict], turn_indices: Set[int], turn_list: List[Tuple[int, int]]
) -> List[Dict]:
    """Strip toolUse/toolResult blocks from specific turns, keeping text content.

    This compacts turns by removing tool execution details while preserving
    the conversation flow (user queries and assistant text responses).

    IMPORTANT: Messages containing 'thinking' or 'redacted_thinking' blocks
    are kept unchanged - Bedrock/Claude requires these blocks to remain unmodified.

    Messages that become empty (tool-result-only messages) are removed entirely.
    """
    # Build set of message indices in the specified turns
    msg_indices_to_strip: Set[int] = set()
    for turn_idx in turn_indices:
        if 0 <= turn_idx < len(turn_list):
            s, e = turn_list[turn_idx]
            msg_indices_to_strip.update(range(s, e))

    result = []

    for msg_idx, msg in enumerate(messages):
        if msg_idx not in msg_indices_to_strip:
            # Keep message as-is
            result.append(msg)
            continue

        # Check if message contains thinking blocks - these MUST remain unchanged
        # Bedrock/Claude models require thinking blocks to be unmodified
        has_thinking = any("thinking" in block or "redacted_thinking" in block for block in msg.get("content", []))

        if has_thinking:
            # Keep message entirely unchanged to preserve thinking blocks
            result.append(msg)
            continue

        # Strip tool blocks from this message (no thinking blocks present)
        new_content = []
        for block in msg.get("content", []):
            # Keep text, images, reasoning - drop toolUse/toolResult
            if "toolUse" not in block and "toolResult" not in block:
                new_content.append(block)

        # Only keep message if it has content after stripping
        if new_content:
            result.append({**msg, "content": new_content})
        # Empty messages (like tool-result-only user messages) are dropped

    return result


def _fix_incomplete_tool_cycles(messages: List[Dict]) -> List[Dict]:
    """Fix ALL incomplete tool cycles in messages.

    Finds any toolUse blocks without matching toolResult and adds
    synthetic toolResult messages to complete the cycles.

    This is critical for imported conversations that may have been
    exported mid-execution or edited manually.
    """
    if not messages:
        return messages

    pending_ids = _get_pending_tool_use_ids(messages)

    if not pending_ids:
        return messages

    # Find which messages contain the pending toolUse blocks
    # Group by the assistant message they came from for proper ordering
    pending_by_msg_idx = {}
    for idx, msg in enumerate(messages):
        if msg["role"] == "assistant":
            msg_pending = []
            for block in msg.get("content", []):
                if "toolUse" in block and block["toolUse"]["toolUseId"] in pending_ids:
                    msg_pending.append(block["toolUse"]["toolUseId"])
            if msg_pending:
                pending_by_msg_idx[idx] = msg_pending

    if not pending_by_msg_idx:
        return messages

    # Build new message list, inserting synthetic toolResults after each orphaned assistant msg
    result = []
    for idx, msg in enumerate(messages):
        result.append(msg)

        if idx in pending_by_msg_idx:
            # Check if next message already has toolResults for these IDs
            next_idx = idx + 1
            if next_idx < len(messages):
                next_msg = messages[next_idx]
                next_result_ids = set()
                if next_msg["role"] == "user":
                    for block in next_msg.get("content", []):
                        if "toolResult" in block:
                            next_result_ids.add(block["toolResult"]["toolUseId"])

                # Only add synthetic results for IDs not already covered
                missing_ids = [tid for tid in pending_by_msg_idx[idx] if tid not in next_result_ids]
            else:
                missing_ids = pending_by_msg_idx[idx]

            if missing_ids:
                # Insert synthetic toolResult message
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": tid,
                                    "status": "success",
                                    "content": [{"text": "[conversation modified - synthetic result]"}],
                                }
                            }
                            for tid in missing_ids
                        ],
                    }
                )
                logger.info(f"Added synthetic toolResult for {len(missing_ids)} orphaned toolUse(s)")

    return result


def _validate_message_structure(messages: List[Dict]) -> Tuple[bool, str]:
    """Validate that messages have proper structure for Bedrock API.

    Checks:
    1. All toolUse have corresponding toolResult
    2. Messages alternate properly (user/assistant)
    3. toolResult immediately follows its toolUse's assistant message

    Returns:
        Tuple of (is_valid, error_message)
    """
    pending_ids = _get_pending_tool_use_ids(messages)

    if pending_ids:
        return False, f"Found {len(pending_ids)} toolUse without toolResult: {pending_ids[:3]}..."

    # Check alternation (allowing for tool cycles)
    for i in range(1, len(messages)):
        prev_role = messages[i - 1]["role"]
        curr_role = messages[i]["role"]

        # Valid transitions: user→assistant, assistant→user
        if prev_role == curr_role:
            # Same role twice is invalid
            return False, f"Invalid message sequence at index {i}: {prev_role}→{curr_role}"

    return True, ""


def _get_active_turn_messages(messages: List[Dict]) -> Tuple[int, List[Dict]]:
    """Find the current active turn (incomplete tool cycle) that must be preserved.

    The active turn is the ongoing conversation cycle where:
    - There's a user message (the current query)
    - Followed by assistant message with toolUse (this tool call)
    - But NO toolResult yet (because we're still executing)

    Returns:
        Tuple of (start_index, list of active turn messages)
    """
    if not messages:
        return -1, []

    # Walk backwards to find the start of the active turn
    # The active turn starts at the last user message that initiated a tool cycle
    active_start = -1

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        if msg["role"] == "user":
            # Check if this is a toolResult message (part of a previous cycle)
            has_tool_result = any("toolResult" in b for b in msg.get("content", []))
            if not has_tool_result:
                # This is a fresh user query - start of active turn
                active_start = i
                break
            # Otherwise keep looking back

    if active_start == -1:
        return -1, []

    # Return all messages from active_start to end
    return active_start, messages[active_start:]


def _get_active_tool_ids(messages: List[Dict]) -> Set[str]:
    """Get tool IDs that are part of the active (current) turn.

    These cannot be dropped as they're currently executing.
    """
    active_start, active_msgs = _get_active_turn_messages(messages)
    if active_start < 0:
        return set()

    active_ids = set()
    for msg in active_msgs:
        for block in msg.get("content", []):
            if "toolUse" in block:
                active_ids.add(block["toolUse"]["toolUseId"])
            elif "toolResult" in block:
                active_ids.add(block["toolResult"]["toolUseId"])

    return active_ids


@tool
def manage_messages(
    action: str,
    path: Optional[str] = None,
    turns: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    role: Optional[str] = None,
    tool_ids: Optional[str] = None,
    tool_name: Optional[str] = None,
    summary_len: Optional[int] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Manage agent conversation history with turn-aware operations.

    Works directly with agent.messages, providing safe operations that maintain
    message integrity (no orphaned toolUse/toolResult).

    Args:
        action: Action to perform:
            - "list": View messages (optionally filtered by role)
            - "list_tools": View all tool calls with IDs
            - "stats": Get conversation statistics
            - "export": Save to JSON file
            - "import": Load from JSON file (replaces current)
            - "drop": Remove complete turns
            - "drop_tools": Remove specific tool calls
            - "compact": Strip toolUse/toolResult from turns (keep text)
            - "clear": Remove all messages
        path: File path for export/import actions
        turns: Comma-separated turn indices for drop/compact (e.g., "0,2,5")
        start: Start turn index for range operations (inclusive)
        end: End turn index for range operations (exclusive)
        role: Filter by role for list action ("user" or "assistant")
        tool_ids: Comma-separated tool IDs to drop (for drop_tools)
        tool_name: Tool name to drop all instances of (for drop_tools)
        summary_len: Max length for message preview in list actions (default: STRANDS_MESSAGE_SUMMARY_LEN env or 80)
        agent: Agent instance (auto-injected)

    Returns:
        Dict with status and content

    Examples:
        # List all turns
        manage_messages(action="list")

        # List only user messages
        manage_messages(action="list", role="user")

        # List all tool calls with IDs
        manage_messages(action="list_tools")

        # Export conversation
        manage_messages(action="export", path="/tmp/chat.json")

        # Import conversation
        manage_messages(action="import", path="/tmp/chat.json")

        # Drop specific turns
        manage_messages(action="drop", turns="0,2")

        # Drop range of turns
        manage_messages(action="drop", start=0, end=5)

        # Drop specific tool calls by ID
        manage_messages(action="drop_tools", tool_ids="tooluse_abc,tooluse_def")

        # Drop all calls to a specific tool
        manage_messages(action="drop_tools", tool_name="shell")

        # Compact turns (strip tool blocks, keep text)
        manage_messages(action="compact")  # auto-compact all but last 3 turns
        manage_messages(action="compact", turns="0,1,2")  # specific turns
        manage_messages(action="compact", start=0, end=5)  # range

        # Clear all
        manage_messages(action="clear")
    """
    if not agent:
        return {"status": "error", "content": [{"text": "Agent not available"}]}

    messages = agent.messages

    if action == "list":
        if not messages:
            return {"status": "success", "content": [{"text": "No messages"}]}

        if role:
            # Filter by role
            filtered = [(i, m) for i, m in enumerate(messages) if m["role"] == role]
            lines = [f"**{len(filtered)} {role} messages:**\n"]
            for i, msg in filtered:
                lines.append(f"  [{i}] {_summarize(msg['content'], summary_len)}")
        else:
            # Show by turns
            turn_list = _parse_turns(messages)
            lines = [f"**{len(turn_list)} turns ({len(messages)} messages):**\n"]

            for turn_idx, (s, e) in enumerate(turn_list):
                lines.append(f"--- Turn {turn_idx} (msgs {s}-{e - 1}) ---")
                for msg_idx in range(s, e):
                    msg = messages[msg_idx]
                    lines.append(f"  [{msg_idx}] {msg['role']}: {_summarize(msg['content'], summary_len)}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    elif action == "list_tools":
        if not messages:
            return {"status": "success", "content": [{"text": "No messages"}]}

        tool_calls = _get_all_tool_calls(messages)
        if not tool_calls:
            return {"status": "success", "content": [{"text": "No tool calls found"}]}

        active_ids = _get_active_tool_ids(messages)

        lines = [f"**{len(tool_calls)} tool calls:**\n"]
        for i, tc in enumerate(tool_calls):
            status_icon = "✅" if tc["has_result"] else "⏳"
            active_marker = " 🔒(active)" if tc["tool_use_id"] in active_ids else ""

            lines.append(f"{i}. {status_icon} **{tc['tool_name']}**{active_marker}")
            lines.append(f"   ID: `{tc['tool_use_id']}`")
            lines.append(f"   Args: {tc['args_preview'][:60]}...")
            if tc["has_result"]:
                lines.append(f"   Result ({tc['result_status']}): {tc['result_preview'][:50]}...")
            lines.append("")

        lines.append("💡 Use `drop_tools` with `tool_ids` or `tool_name` to remove specific calls")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    elif action == "stats":
        if not messages:
            return {"status": "success", "content": [{"text": "No messages"}]}

        turn_list = _parse_turns(messages)
        user_count = sum(1 for m in messages if m["role"] == "user")
        asst_count = sum(1 for m in messages if m["role"] == "assistant")

        # Count content types
        counts = {"text": 0, "toolUse": 0, "toolResult": 0, "image": 0, "reasoningContent": 0}
        for msg in messages:
            for block in msg["content"]:
                for key in counts:
                    if key in block:
                        counts[key] += 1

        # Check for pending tool cycles
        pending_ids = _get_pending_tool_use_ids(messages)

        # Count tools by name
        tool_calls = _get_all_tool_calls(messages)
        tool_counts: Dict[str, int] = {}
        for tc in tool_calls:
            tool_counts[tc["tool_name"]] = tool_counts.get(tc["tool_name"], 0) + 1

        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
        tools_summary = ", ".join(f"{name}({count})" for name, count in top_tools)

        stats = f"""**Conversation Statistics:**
• Turns: {len(turn_list)}
• Messages: {len(messages)} (user: {user_count}, assistant: {asst_count})
• Content: {counts["text"]} text, {counts["toolUse"]} toolUse, {counts["toolResult"]} toolResult, \
{counts["image"]} images, {counts["reasoningContent"]} reasoning
• Top tools: {tools_summary}
• Pending toolUse (no result): {len(pending_ids)}"""

        if pending_ids:
            stats += f"\n⚠️ Warning: {len(pending_ids)} orphaned toolUse blocks detected"

        return {"status": "success", "content": [{"text": stats}]}

    elif action == "export":
        if not path:
            return {"status": "error", "content": [{"text": "Required: path parameter"}]}

        export_path = Path(path).expanduser()
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, default=str)

        turn_count = len(_parse_turns(messages))
        pending_count = len(_get_pending_tool_use_ids(messages))

        result_text = f"✅ Exported {turn_count} turns ({len(messages)} messages) to {export_path}"
        if pending_count > 0:
            result_text += f"\n⚠️ Note: {pending_count} pending toolUse blocks included (active tool cycle)"

        return {"status": "success", "content": [{"text": result_text}]}

    elif action == "import":
        if not path:
            return {"status": "error", "content": [{"text": "Required: path parameter"}]}

        import_path = Path(path).expanduser()
        if not import_path.exists():
            return {"status": "error", "content": [{"text": f"File not found: {import_path}"}]}

        with open(import_path, "r", encoding="utf-8") as f:
            imported = json.load(f)

        if not isinstance(imported, list):
            return {"status": "error", "content": [{"text": "Invalid format: expected list"}]}

        # CRITICAL FIX: Validate and fix tool cycles in imported messages
        # This prevents "toolUse without toolResult" errors from Bedrock
        original_count = len(imported)
        imported = _fix_incomplete_tool_cycles(imported)
        fixed_count = len(imported) - original_count

        # Validate the fixed messages
        is_valid, error_msg = _validate_message_structure(imported)
        if not is_valid:
            return {
                "status": "error",
                "content": [{"text": f"Invalid message structure after fix attempt: {error_msg}"}],
            }

        # CRITICAL: Preserve the active turn
        active_start, active_turn_msgs = _get_active_turn_messages(messages)

        # Replace messages with imported + active turn
        messages.clear()
        messages.extend(imported)
        messages.extend(active_turn_msgs)

        turn_count = len(_parse_turns(imported))
        result_text = f"✅ Imported {turn_count} turns ({len(imported)} messages) from {import_path}"
        if fixed_count > 0:
            result_text += f"\n🔧 Fixed {fixed_count} incomplete tool cycle(s) with synthetic results"
        result_text += ", preserved active turn"

        return {"status": "success", "content": [{"text": result_text}]}

    elif action == "drop":
        if not messages:
            return {"status": "success", "content": [{"text": "No messages to drop"}]}

        # CRITICAL: Identify and preserve the active turn (current tool execution cycle)
        active_start, active_turn_msgs = _get_active_turn_messages(messages)

        # Only consider messages BEFORE the active turn for dropping
        droppable_messages = messages[:active_start] if active_start > 0 else []

        if not droppable_messages:
            return {
                "status": "success",
                "content": [{"text": "No droppable messages (only active turn exists)"}],
            }

        turn_list = _parse_turns(droppable_messages)
        if not turn_list:
            return {"status": "success", "content": [{"text": "No complete turns found to drop"}]}

        # Parse turn indices to drop
        drop_indices = set()

        if turns:
            for part in turns.split(","):
                try:
                    drop_indices.add(int(part.strip()))
                except ValueError:
                    return {"status": "error", "content": [{"text": f"Invalid turn index: {part}"}]}

        if start is not None and end is not None:
            drop_indices.update(range(start, end))
        elif start is not None:
            drop_indices.update(range(start, len(turn_list)))
        elif end is not None:
            drop_indices.update(range(0, end))

        if not drop_indices:
            return {
                "status": "error",
                "content": [{"text": "Specify turns='0,1,2' or start/end parameters"}],
            }

        # Collect message indices to drop (only from droppable messages)
        msg_indices_to_drop: Set[int] = set()
        for turn_idx in drop_indices:
            if 0 <= turn_idx < len(turn_list):
                s, e = turn_list[turn_idx]
                msg_indices_to_drop.update(range(s, e))

        # Keep messages not in drop set (from droppable portion only)
        kept = [m for i, m in enumerate(droppable_messages) if i not in msg_indices_to_drop]

        # Fix any incomplete tool cycles in kept portion
        kept = _fix_incomplete_tool_cycles(kept)

        # CRITICAL: Re-append the active turn messages to maintain conversation integrity
        kept.extend(active_turn_msgs)

        # Update agent.messages directly
        messages.clear()
        messages.extend(kept)

        new_turn_count = len(_parse_turns(kept[: len(kept) - len(active_turn_msgs)]))
        return {
            "status": "success",
            "content": [
                {
                    "text": f"✅ Dropped {len(drop_indices)} turns ({len(msg_indices_to_drop)} messages)\n"
                    f"Remaining: {new_turn_count} completed turns + active turn ({len(kept)} messages total)"
                }
            ],
        }

    elif action == "drop_tools":
        if not messages:
            return {"status": "success", "content": [{"text": "No messages"}]}

        if not tool_ids and not tool_name:
            return {
                "status": "error",
                "content": [{"text": "Required: tool_ids (comma-separated) or tool_name parameter"}],
            }

        # Get active tool IDs that cannot be dropped
        active_ids = _get_active_tool_ids(messages)

        # Build set of tool IDs to drop
        ids_to_drop = set()

        if tool_ids:
            for tid in tool_ids.split(","):
                tid = tid.strip()
                if tid in active_ids:
                    return {"status": "error", "content": [{"text": f"Cannot drop active tool: {tid}"}]}
                ids_to_drop.add(tid)

        if tool_name:
            # Find all tool calls with matching name
            tool_calls = _get_all_tool_calls(messages)
            for tc in tool_calls:
                if tc["tool_name"] == tool_name:
                    if tc["tool_use_id"] in active_ids:
                        continue  # Skip active tools
                    ids_to_drop.add(tc["tool_use_id"])

        if not ids_to_drop:
            return {
                "status": "success",
                "content": [{"text": f"No droppable tool calls found{' for ' + tool_name if tool_name else ''}"}],
            }

        # CRITICAL: Preserve active turn
        active_start, active_turn_msgs = _get_active_turn_messages(messages)
        droppable_messages = messages[:active_start] if active_start > 0 else []

        if not droppable_messages:
            return {"status": "success", "content": [{"text": "No droppable messages (only active turn exists)"}]}

        # Remove tool blocks
        modified = _remove_tool_blocks(droppable_messages, ids_to_drop)

        # Fix any incomplete cycles created by removal
        modified = _fix_incomplete_tool_cycles(modified)

        # Re-append active turn
        modified.extend(active_turn_msgs)

        # Update messages
        messages.clear()
        messages.extend(modified)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"✅ Dropped {len(ids_to_drop)} tool call(s)\n"
                    f"Removed: {', '.join(list(ids_to_drop)[:5])}{'...' if len(ids_to_drop) > 5 else ''}"
                }
            ],
        }

    elif action == "compact":
        # Strip toolUse/toolResult from older turns to reduce context size
        if not messages:
            return {"status": "success", "content": [{"text": "No messages to compact"}]}

        # CRITICAL: Preserve active turn
        active_start, active_turn_msgs = _get_active_turn_messages(messages)
        droppable_messages = messages[:active_start] if active_start > 0 else []

        if not droppable_messages:
            return {
                "status": "success",
                "content": [{"text": "No droppable messages (only active turn exists)"}],
            }

        turn_list = _parse_turns(droppable_messages)
        if not turn_list:
            return {"status": "success", "content": [{"text": "No complete turns found to compact"}]}

        # Parse turn indices to compact
        compact_indices = set()

        if turns:
            for part in turns.split(","):
                try:
                    compact_indices.add(int(part.strip()))
                except ValueError:
                    return {"status": "error", "content": [{"text": f"Invalid turn index: {part}"}]}

        if start is not None and end is not None:
            compact_indices.update(range(start, end))
        elif start is not None:
            compact_indices.update(range(start, len(turn_list)))
        elif end is not None:
            compact_indices.update(range(0, end))

        # Default: compact all turns except the last few (keep recent context)
        if not compact_indices and not turns and start is None and end is None:
            # By default, compact all but the last 3 turns
            keep_recent = 3
            if len(turn_list) > keep_recent:
                compact_indices = set(range(len(turn_list) - keep_recent))
            else:
                return {
                    "status": "success",
                    "content": [{"text": f"Only {len(turn_list)} turns - nothing to compact (keeping last 3)"}],
                }

        if not compact_indices:
            return {
                "status": "error",
                "content": [
                    {"text": "Specify turns='0,1,2' or start/end parameters, or run without params to auto-compact"}
                ],
            }

        # Count tool blocks before compaction
        tool_blocks_before = sum(
            1 for m in droppable_messages for b in m.get("content", []) if "toolUse" in b or "toolResult" in b
        )

        # Strip tool blocks from specified turns
        compacted = _strip_tool_blocks_from_turns(droppable_messages, compact_indices, turn_list)

        # Count tool blocks after compaction
        tool_blocks_after = sum(
            1 for m in compacted for b in m.get("content", []) if "toolUse" in b or "toolResult" in b
        )

        # Re-append active turn
        compacted.extend(active_turn_msgs)

        # Update messages
        messages.clear()
        messages.extend(compacted)

        removed_blocks = tool_blocks_before - tool_blocks_after
        return {
            "status": "success",
            "content": [
                {
                    "text": f"✅ Compacted {len(compact_indices)} turns\n"
                    f"Removed {removed_blocks} tool blocks (toolUse + toolResult)\n"
                    f"Messages: {len(droppable_messages)} → {len(compacted) - len(active_turn_msgs)} + active turn"
                }
            ],
        }

    elif action == "clear":
        # CRITICAL: Preserve the active turn to maintain conversation integrity
        active_start, active_turn_msgs = _get_active_turn_messages(messages)

        # Count what we're clearing (everything before active turn)
        droppable_count = active_start if active_start > 0 else 0
        droppable_messages = messages[:active_start] if active_start > 0 else []
        turn_count = len(_parse_turns(droppable_messages))

        # Clear and re-add only the active turn
        messages.clear()
        messages.extend(active_turn_msgs)

        return {
            "status": "success",
            "content": [{"text": f"✅ Cleared {turn_count} turns ({droppable_count} messages), preserved active turn"}],
        }

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. "
                    "Valid: list, list_tools, stats, export, import, drop, drop_tools, compact, clear"
                }
            ],
        }
