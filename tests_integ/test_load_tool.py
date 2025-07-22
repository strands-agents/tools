import pytest
from strands import Agent
from strands_tools import load_tool


@pytest.fixture
def temporary_tool(tmp_path):
    # Create a simple tool in a temporary directory
    tools_dir = tmp_path / ".strands" / "tools"
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
    """Test dynamic tool loading functionality using semantic query."""
    tool_name, tool_path, tools_dir_str = temporary_tool
    agent = Agent(tools=[load_tool])

    response = agent(f"Please load the Python tool from {tool_path} add_numbers and calculate 1 + 2.")

    response_text = str(response).lower()
    assert "loaded" in response_text and "success" in response_text
    assert "add_number" in response_text and "3" in response_text
