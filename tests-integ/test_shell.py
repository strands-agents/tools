import json
import os
from unittest.mock import patch

import pytest
from strands import Agent


def test_shell_command_non_interactive_mode():
    """Test shell command execution in non-interactive mode."""
    # Skip on Windows as shell tool doesn't work under Windows Os
    if os.name == "nt":
        pytest.skip("shell tool not supported on Windows")

    from strands_tools import shell

    agent = Agent(tools=[shell])

    res = agent.tool.shell(command="echo 'hello shell'", non_interactive_mode=True)
    assert res["status"] == "success"
    output_content = ""
    for item in res.get("content", []):
        if "Output:" in item.get("text", ""):
            output_content = item["text"]
            break

    assert "hello shell" in output_content, "The expected output was not found in the result."


@patch("strands_tools.utils.user_input.get_user_input")
def test_shell_command_interactive_mode_confirm(mock_get_user_input):
    """Test shell command execution in interactive mode with user confirmation."""
    if os.name == "nt":
        pytest.skip("shell tool not supported on Windows")

    # Configure the mock to return 'y' to simulate the user confirming the command.
    mock_get_user_input.return_value = "y"

    from strands_tools import shell

    agent = Agent(tools=[shell])

    res = agent.tool.shell(command="echo 'hello interactive shell'", non_interactive_mode=False)

    mock_get_user_input.assert_called_once()

    assert res["status"] == "success"

    output_content = ""
    for item in res.get("content", []):
        if "Output:" in item.get("text", ""):
            output_content = item["text"]
            break

    assert "hello interactive shell" in output_content, "The expected output was not found in the result."


@patch("strands_tools.utils.user_input.get_user_input")
def test_shell_command_interactive_mode_cancel(mock_get_user_input):
    """Test shell command cancellation in interactive mode."""
    if os.name == "nt":
        pytest.skip("shell tool not supported on Windows")

    # Configure the mock to return 'n' to simulate the user cancelling the command.
    mock_get_user_input.return_value = "n"

    from strands_tools import shell

    agent = Agent(tools=[shell])

    res = agent.tool.shell(command="echo 'this should not run'", non_interactive_mode=False)

    mock_get_user_input.assert_called_once()

    assert res["status"] == "error"

    assert "cancelled by user" in json.dumps(res.get("content", [])).lower(), "The cancellation message was not found."
