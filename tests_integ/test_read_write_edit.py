import os
import re
from unittest.mock import patch

import pytest
from strands import Agent
from strands_tools import editor, file_read, file_write


@pytest.fixture
def agent():
    """Agent with file read, write, and editor tools."""
    return Agent(tools=[file_write, file_read, editor])


@pytest.fixture(autouse=True)
def bypass_tool_consent_env():
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
        yield


def extract_code_content(response):
    """Helper function to extract code block content from LLM output."""
    match = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", str(response), re.DOTALL)
    return match.group(1) if match else str(response)


def test_semantic_write_read_edit_workflow(agent, tmp_path):
    """Test complete semantic workflow: write -> read -> edit -> verify."""
    file_path = tmp_path / "semantic_test.txt"
    initial_content = "Hello world from integration test!"

    # 1. Write file
    write_response = agent(f"Write '{initial_content}' to file `{file_path}`")
    assert "success" in str(write_response).lower() or "written" in str(write_response).lower()

    # 2. Read file back using both agent & reading file directly
    read_response = agent(f"Read the contents of file `{file_path}`")
    content = extract_code_content(read_response)
    assert initial_content in content

    with open(file_path, "r") as f:
        raw_content = f.read()
    assert initial_content in raw_content

    # 3. Replace text
    edit_response = agent(f"In file `{file_path}`, replace 'Hello' with 'Hi' - write/edit nothing else")
    assert "success" in str(edit_response).lower() or "replaced" in str(edit_response).lower()

    # 4. Verify
    verify_response = agent(
        f"Show me the contents of `{file_path}` - do not respond with anything but the exact content in the file"
    )
    final_content = extract_code_content(verify_response)
    assert "Hi world" in final_content
    assert "Hello" not in final_content


def test_semantic_python_file_creation(agent, tmp_path):
    """Test creating and modifying Python code semantically."""
    file_path = tmp_path / "test_script.py"

    # 1. Create Python file
    create_response = agent(f"Create a Python file at `{file_path}` with a function that prints 'Hello World'")
    assert "success" in str(create_response).lower() or "created" in str(create_response).lower()

    # 2. Read and verify
    read_response = agent(f"Show me the Python code in `{file_path}`")
    content = str(read_response)
    assert "def" in content and "print" in content and "Hello World" in content

    # 3. Modify the function
    modify_response = agent(f"""
        In `{file_path}`, change the print statement to say 'Hi there!' instead.
        If successful respond ONLY with 'PASS',
        else, respond with ONLY with 'FAIL'
    """)
    assert 'PASS' in str(modify_response)

    # 4. Verify modification
    final_response = agent(f"Read `{file_path}` and show me the code")
    final_content = str(final_response)
    assert "Hi there!" in final_content


def test_semantic_search_and_replace(agent, tmp_path):
    """Test semantic search and replace operations."""
    file_path = tmp_path / "config.txt"

    # 1. Create config file
    agent(f"Create a config file at `{file_path}` with settings: debug=true, port=8080, host=localhost")

    # 2. Change a specific setting
    agent(f"In `{file_path}`, change the port from 8080 to 3000")

    # 3. Verify the change
    verify_response = agent(f"What is the port setting in `{file_path}`?")
    assert "3000" in str(verify_response)

    # 4. Final check
    final_response = agent(f"Show me all settings in `{file_path}`")
    final_content = str(final_response)
    assert "3000" in final_content
