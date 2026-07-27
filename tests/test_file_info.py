"""
Tests for file_info utilities.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from strands_tools.utils.file_info import get_file_info, format_file_info, FileInfo


def test_get_file_info_for_file():
    """Test getting file info for a regular file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        file_info = get_file_info(temp_path)
        
        assert "path" in file_info
        assert "is_dir" in file_info
        assert "size" in file_info
        assert "modified_at" in file_info
        
        assert file_info["is_dir"] is False
        assert file_info["size"] > 0
        assert Path(file_info["path"]).exists()
        
        # Check that modified_at is a valid ISO format
        datetime.fromisoformat(file_info["modified_at"])
        
    finally:
        os.unlink(temp_path)


def test_get_file_info_for_directory():
    """Test getting file info for a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_info = get_file_info(temp_dir)
        
        assert file_info["is_dir"] is True
        assert file_info["size"] >= 0
        assert Path(file_info["path"]).exists()
        assert Path(file_info["path"]).is_dir()


def test_get_file_info_nonexistent_file():
    """Test that get_file_info raises FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        get_file_info("/nonexistent/path/to/file.txt")


def test_format_file_info():
    """Test formatting file info as string."""
    file_info: FileInfo = {
        "path": "/tmp/test.txt",
        "is_dir": False,
        "size": 1024,
        "modified_at": "2024-01-01T12:00:00"
    }
    
    formatted = format_file_info(file_info)
    
    assert "Path: /tmp/test.txt" in formatted
    assert "Type: file" in formatted
    assert "Size: 1.00 KB" in formatted
    assert "Modified: 2024-01-01T12:00:00" in formatted


def test_format_file_info_directory():
    """Test formatting directory info as string."""
    file_info: FileInfo = {
        "path": "/tmp/testdir",
        "is_dir": True,
        "size": 0,
        "modified_at": "2024-01-01T12:00:00"
    }
    
    formatted = format_file_info(file_info)
    
    assert "Path: /tmp/testdir" in formatted
    assert "Type: directory" in formatted


def test_file_info_has_correct_path():
    """Test that file info contains the correct absolute path."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        file_info = get_file_info(temp_path)
        
        # Check that path is absolute
        assert os.path.isabs(file_info["path"])
        
        # Check that path is resolved
        assert file_info["path"] == str(Path(temp_path).resolve())
        
    finally:
        os.unlink(temp_path)


def test_file_info_timestamp_format():
    """Test that timestamp is in ISO format."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        file_info = get_file_info(temp_path)
        
        # Should be able to parse as ISO format
        modified_time = datetime.fromisoformat(file_info["modified_at"])
        
        # Should be recent (within last minute)
        time_diff = datetime.now() - modified_time
        assert abs(time_diff.total_seconds()) < 60
        
    finally:
        os.unlink(temp_path)


def test_get_file_info_with_different_sizes():
    """Test that file size is correctly reported."""
    test_content = "A" * 1000  # 1000 bytes
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        file_info = get_file_info(temp_path)
        
        # Size should be close to 1000 bytes
        assert file_info["size"] == 1000
        
    finally:
        os.unlink(temp_path)


def test_format_file_info_with_large_file():
    """Test formatting info for a large file."""
    file_info: FileInfo = {
        "path": "/tmp/large_file.bin",
        "is_dir": False,
        "size": 1024 * 1024 * 10,  # 10 MB
        "modified_at": "2024-01-01T12:00:00"
    }
    
    formatted = format_file_info(file_info)
    
    # Should show size in KB
    assert "Size: 10240.00 KB" in formatted

