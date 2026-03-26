"""
Tests for the glob tool.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from strands import Agent

from strands_tools import glob


@pytest.fixture
def agent():
    """Create an agent with the glob tool loaded."""
    return Agent(tools=[glob])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_glob_basic_match(tmp_path):
    """Test basic glob matching."""
    # Create test files
    (tmp_path / "file1.json").touch()
    (tmp_path / "file2.json").touch()

    result = glob.glob(pattern="*.json", path=str(tmp_path))

    assert "file1.json" in result
    assert "file2.json" in result


def test_glob_recursive_pattern(tmp_path):
    """Test recursive glob with **."""
    # Create nested structure
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (tmp_path / "file1.py").touch()
    (subdir / "file2.py").touch()

    result = glob.glob(pattern="**/*.py", path=str(tmp_path))

    assert "file1.py" in result
    assert "file2.py" in result


def test_glob_no_matches(tmp_path):
    """Test when no files match."""
    result = glob.glob(pattern="*.nonexistent", path=str(tmp_path))

    assert "No files found" in result


def test_glob_invalid_path():
    """Test handling of non-existent path."""
    result = glob.glob(pattern="*.py", path="/nonexistent/path")

    # Non-existent paths return no matches, not an error
    assert "No files found" in result


def test_glob_filters_directories(tmp_path):
    """Test that directories are excluded from results."""
    (tmp_path / "file.py").touch()
    (tmp_path / "subdir").mkdir()

    result = glob.glob(pattern="*", path=str(tmp_path))

    assert "file.py" in result
    assert "subdir" not in result or (tmp_path / "subdir" / "file.py").exists()


def test_glob_default_path_is_cwd(monkeypatch):
    """Test that default path is current working directory."""
    # Create a temporary directory and change to it
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "file.txt").touch()

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        result = glob.glob(pattern="*.txt")

        assert "file.txt" in result


def test_glob_sorting_by_mtime(tmp_path):
    """Test that results are sorted by modification time."""
    import time

    old_file = tmp_path / "old.py"
    old_file.touch()

    time.sleep(0.01)  # Ensure different mtime

    new_file = tmp_path / "new.py"
    new_file.touch()

    result = glob.glob(pattern="*.py", path=str(tmp_path))
    lines = result.split("\n")

    # new.py should appear before old.py (sorted by mtime desc)
    # Find which line contains which file
    new_idx = next((i for i, line in enumerate(lines) if "new.py" in line), -1)
    old_idx = next((i for i, line in enumerate(lines) if "old.py" in line), -1)

    assert new_idx >= 0 and old_idx >= 0
    assert new_idx < old_idx


def test_glob_via_agent(agent, tmp_path):
    """Test glob via agent interface."""
    (tmp_path / "result.py").touch()

    result = agent.tool.glob(pattern="*.py", path=str(tmp_path))

    result_text = extract_result_text(result)
    assert "result.py" in result_text


def test_glob_absolute_paths(tmp_path):
    """Test that results contain absolute paths."""
    (tmp_path / "file.py").touch()

    result = glob.glob(pattern="*.py", path=str(tmp_path))

    # Check that paths are absolute
    assert str(tmp_path) in result or result.startswith("/")


def test_glob_empty_directory(tmp_path):
    """Test glob in empty directory."""
    result = glob.glob(pattern="*.py", path=str(tmp_path))

    assert "No files found" in result


def test_glob_complex_pattern(tmp_path):
    """Test complex glob patterns."""
    # Create various files
    (tmp_path / "test_file.py").touch()
    (tmp_path / "other_file.txt").touch()
    (tmp_path / "another_test.py").touch()

    result = glob.glob(pattern="test*.py", path=str(tmp_path))

    assert "test_file.py" in result
    assert "other_file.txt" not in result


def test_glob_exception_handling(tmp_path):
    """Test generic exception handling."""
    # Create a file and then mock glob to raise exception
    with patch.object(Path, "glob", side_effect=Exception("Unexpected error")):
        result = glob.glob(pattern="*.py", path=str(tmp_path))

    assert "Error" in result
    assert "Unexpected error" in result


def test_glob_path_resolution(tmp_path):
    """Test that relative paths are resolved correctly."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file.py").touch()

    result = glob.glob(pattern="**/*.py", path=str(tmp_path))

    # Should contain resolved absolute path
    assert "subdir" in result
    assert "file.py" in result
