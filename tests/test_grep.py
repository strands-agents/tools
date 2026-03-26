"""
Tests for the grep tool.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import grep


@pytest.fixture
def agent():
    """Create an agent with the grep tool loaded."""
    return Agent(tools=[grep])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_grep_basic_search():
    """Test basic pattern search."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "file1.py\nfile2.py\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = grep.grep(pattern="test_pattern")

    # Should return file paths
    assert "file1.py" in result
    assert "file2.py" in result


def test_grep_with_path():
    """Test search in specific directory."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "src/module.py\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        grep.grep(pattern="pattern", path="src/")

        # Verify path was passed to ripgrep
        args = mock_run.call_args[0][0]
        assert "src/" in args
        assert "pattern" in args
        assert "-l" in args  # files-with-matches flag


def test_grep_with_include_filter():
    """Test file filtering with include parameter."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "test.py\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        grep.grep(pattern="pattern", include="*.py")

        # Verify --glob flag was passed
        args = mock_run.call_args[0][0]
        assert "--glob" in args
        assert "*.py" in args


def test_grep_no_matches():
    """Test when no matches found."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = grep.grep(pattern="nonexistent")

    assert "No files found" in result


def test_grep_error():
    """Test error handling."""
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stderr = "Invalid regex pattern"
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        result = grep.grep(pattern="invalid[pattern")

    assert "Error searching" in result


def test_grep_ripgrep_not_found():
    """Test error when ripgrep not installed."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = grep.grep(pattern="pattern")

    assert "ripgrep" in result.lower()
    assert "not found" in result.lower()


def test_grep_timeout():
    """Test timeout handling."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
        result = grep.grep(pattern="pattern")

    assert "timed out" in result.lower()


def test_grep_sorting_by_mtime():
    """Test that results are sorted by modification time."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "old.py\nnew.py\n"
    mock_result.stderr = ""

    # Mock Path.stat to return different mtimes
    def mock_stat(self):
        mock_stat_result = MagicMock()
        if "new.py" in str(self):
            mock_stat_result.st_mtime = 2000
        else:
            mock_stat_result.st_mtime = 1000
        return mock_stat_result

    with patch("subprocess.run", return_value=mock_result):
        with patch.object(Path, "stat", mock_stat):
            result = grep.grep(pattern="pattern")

            # new.py should appear before old.py (sorted by mtime desc)
            lines = result.split("\n")
            assert any("new.py" in line for line in lines[:1])


def test_grep_via_agent(agent):
    """Test grep via agent interface."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "result.py\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = agent.tool.grep(pattern="test")

    result_text = extract_result_text(result)
    assert "result.py" in result_text


def test_grep_empty_stdout():
    """Test handling of empty stdout with success code."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = grep.grep(pattern="pattern")

    # Should return empty string or handle gracefully
    assert isinstance(result, str)


def test_grep_exception_handling():
    """Test generic exception handling."""
    with patch("subprocess.run", side_effect=Exception("Unexpected error")):
        result = grep.grep(pattern="pattern")

    assert "Error" in result
    assert "Unexpected error" in result
