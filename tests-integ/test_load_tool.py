import re
import sys
from unittest.mock import patch

import pytest
from strands import Agent


@pytest.fixture
def temporary_tool(tmp_path):
    # Create a simple tool in a temporary directory provided by pytest's tmp_path fixture
    tools_dir = tmp_path / ".strand" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    tools_dir_str = str(tools_dir)

    tool_code = '''from strands import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
    tool_name = "add_numbers"
    tool_path = tools_dir / f"{tool_name}.py"
    tool_path.write_text(tool_code)

    yield tool_name, str(tool_path), tools_dir_str


def test_load_tool_functionality(temporary_tool):
    """Test dynamic tool loading functionality using the temporary_tool fixture."""
    from strands_tools import load_tool

    tool_name, tool_path, tools_dir_str = temporary_tool

    agent = Agent(tools=[load_tool])

    # Use patch.object to temporarily modify sys.path, ensuring the new tool
    # can be found without permanently altering the system path.
    with patch.object(sys, "path", [tools_dir_str] + sys.path):
        res = agent.tool.load_tool(name=tool_name, path=str(tool_path))
        assert res["status"] == "success", f"Tool loading failed with content: {res.get('content')}"

        assert len(res.get("content", [])) > 0
        success_message = res["content"][0].get("text", "")
        match = re.search(r"loaded\s+successfully", success_message, re.IGNORECASE)
        assert match is not None, f"Could not find 'loaded successfully' in success message: {success_message}"
