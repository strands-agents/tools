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

    response = agent(f"""
    Please load the Python tool from {tool_path} add_numbers and calculate 1 + 2.
    If you are unable to load the tool, stop and immediately respond with `FAIL`
    
    If successful return exactly the following `SUM: _` where _ is replaced with the sum. For example `SUM: 27`,
    else, return EXACTLY `FAIL` and nothing else. 
    """)

    assert 'add_numbers' in agent.tool_names
    assert "sum: 3" in str(response).lower()
