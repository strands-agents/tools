"""
Tests for the file_read tool using the Agent interface.
"""

import io
import os
import tempfile
import unittest.mock

import pytest
from rich.console import Console
from strands import Agent

from strands_tools import file_read


@pytest.fixture
def agent():
    """Create an agent with the file_read tool loaded."""
    return Agent(tools=[file_read])


@pytest.fixture
def temp_test_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
        temp.write("Line 1: Test content\nLine 2: More content\nLine 3: Final line")
        temp_name = temp.name

    yield temp_name

    # Clean up after test
    if os.path.exists(temp_name):
        os.remove(temp_name)


@pytest.fixture
def temp_bracket_file():
    """Create a temporary file containing bracket characters."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
        temp.write("Line [1]: Bracket test\nAnother line with text")
        temp_name = temp.name

    yield temp_name

    if os.path.exists(temp_name):
        os.remove(temp_name)


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1_path = os.path.join(temp_dir, "test1.txt")
        file2_path = os.path.join(temp_dir, "test2.md")

        with open(file1_path, "w") as f:
            f.write("Content of test1.txt")

        with open(file2_path, "w") as f:
            f.write("# Test Markdown\nContent of test2.md")

        yield temp_dir


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_file_read_tool_direct_view(temp_test_file):
    """Test direct invocation of the file_read tool in view mode."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_test_file, "mode": "view"},
    }

    # Call the file_read function directly
    result = file_read.file_read(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert "Content of" in result["content"][0]["text"]
    assert "Line 1: Test content" in result["content"][0]["text"]


def test_file_read_tool_direct_find(temp_test_dir):
    """Test direct invocation of the file_read tool in find mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": os.path.join(temp_test_dir, "*.txt"), "mode": "find"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert "Found " in result["content"][0]["text"]
    assert "test1.txt" in result["content"][0]["text"]
    assert "test2.md" not in result["content"][0]["text"]  # Should not find the .md file


def test_file_read_tool_direct_lines(temp_test_file):
    """Test direct invocation of the file_read tool in lines mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "path": temp_test_file,
            "mode": "lines",
            "start_line": 1,  # This is the second line (0-indexed)
            "end_line": 2,  # Up to but not including this line
        },
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert "Line 2: More content" in result["content"][0]["text"]
    assert "Line 1: Test content" not in result["content"][0]["text"]
    assert "Line 3: Final line" not in result["content"][0]["text"]


def test_file_read_tool_direct_search(temp_test_file):
    """Test direct invocation of the file_read tool in search mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_test_file, "mode": "search", "search_pattern": "More"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    # The search result includes rich text formatting, so we just check for basic content
    assert "More" in str(result["content"])


def test_file_read_tool_direct_stats(temp_test_file):
    """Test direct invocation of the file_read tool in stats mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_test_file, "mode": "stats"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert "line_count" in result["content"][0]["text"]
    assert "size_bytes" in result["content"][0]["text"]


def test_file_read_tool_direct_chunk(temp_test_file):
    """Test direct invocation of the file_read tool in chunk mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "path": temp_test_file,
            "mode": "chunk",
            "chunk_size": 10,
            "chunk_offset": 0,
        },
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert len(result["content"][0]["text"]) <= 10  # Chunk size is 10 bytes


def test_file_read_preview_mode(temp_test_file):
    """Test file_read tool in preview mode."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_test_file, "mode": "preview"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "success"
    assert "Preview:" in result["content"][0]["text"]
    assert "Total Lines:" in result["content"][0]["text"]


def test_file_read_via_agent(agent, temp_test_file):
    """Test file_read via the agent interface."""
    result = agent.tool.file_read(path=temp_test_file, mode="view")

    result_text = extract_result_text(result)
    assert "Content of" in result_text
    assert "Line 1: Test content" in result_text


def test_file_read_multiple_files(temp_test_dir, agent):
    """Test file_read with multiple files using comma-separated paths."""
    file1_path = os.path.join(temp_test_dir, "test1.txt")
    file2_path = os.path.join(temp_test_dir, "test2.md")

    result = agent.tool.file_read(path=f"{file1_path},{file2_path}", mode="find")

    result_text = extract_result_text(result)
    assert "Found 2 files" in result_text
    assert "test1.txt" in result_text
    assert "test2.md" in result_text


def test_file_read_error_handling_file_not_found():
    """Test file_read error handling when file not found."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": "/path/that/does/not/exist.txt", "mode": "view"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "error"
    assert "No files found" in result["content"][0]["text"]


def test_file_read_error_handling_missing_parameters():
    """Test file_read error handling with missing parameters."""
    # Missing path parameter
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"mode": "view"}}

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "error"
    assert "path parameter is required" in result["content"][0]["text"]

    # Missing mode parameter
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"path": "/some/path"}}

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "error"
    assert "mode parameter is required" in result["content"][0]["text"]


def test_find_files_function():
    """Test the find_files utility function directly."""
    mock_console = unittest.mock.Mock()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1_path = os.path.join(temp_dir, "test1.txt")
        file2_path = os.path.join(temp_dir, "test2.md")
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        file3_path = os.path.join(subdir, "test3.txt")

        with open(file1_path, "w") as f:
            f.write("Content 1")
        with open(file2_path, "w") as f:
            f.write("Content 2")
        with open(file3_path, "w") as f:
            f.write("Content 3")

        # Test with direct path
        files = file_read.find_files(mock_console, file1_path)
        assert len(files) == 1
        assert files[0] == file1_path

        # Test with pattern - will find both txt files because recursive is True by default
        files = file_read.find_files(mock_console, os.path.join(temp_dir, "*.txt"))
        assert len(files) == 2
        assert any(os.path.basename(f) == "test1.txt" for f in files)
        assert any(os.path.basename(f) == "test3.txt" for f in files)

        # Test with recursive
        files = file_read.find_files(mock_console, temp_dir, recursive=True)
        assert len(files) == 3

        # Test with non-recursive
        files = file_read.find_files(mock_console, temp_dir, recursive=False)
        assert len(files) == 2  # Should not find test3.txt in subdir


def test_split_path_list_function():
    """Test the split_path_list utility function."""
    paths = "path1.txt,path2.md, path3.py"
    result = file_read.split_path_list(paths)
    assert len(result) == 3
    assert result[0] == "path1.txt"
    assert result[1] == "path2.md"
    assert result[2] == "path3.py"


def test_create_rich_panel_function():
    """Test the create_rich_panel utility function."""
    panel = file_read.create_rich_panel("Content", "Title", "file.py")
    assert panel is not None


def test_read_file_lines_handles_brackets(temp_bracket_file):
    """Ensure read_file_lines can handle lines with '[' safely."""
    console = Console(file=io.StringIO())

    lines = file_read.read_file_lines(console, temp_bracket_file, 0, 1)

    assert "[1]" in lines[0]


def test_read_file_chunk_handles_brackets(temp_bracket_file):
    """Ensure read_file_chunk can handle content with '[' safely."""
    console = Console(file=io.StringIO())

    content = file_read.read_file_chunk(console, temp_bracket_file, chunk_size=20)

    assert "[1]" in content


def test_file_read_error_message_brackets():
    """Errors containing '[' characters should not raise markup issues."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": "missing[file].txt", "mode": "view"},
    }

    result = file_read.file_read(tool=tool_use)

    assert result["status"] == "error"
