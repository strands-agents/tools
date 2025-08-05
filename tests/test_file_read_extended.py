"""
Extended tests for file_read.py to improve coverage from 57% to 80%+.
"""

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.strands_tools.file_read import (
    create_diff,
    create_document_block,
    create_document_response,
    create_rich_panel,
    detect_format,
    file_read,
    find_files,
    get_file_stats,
    read_file_chunk,
    read_file_lines,
    search_file,
    split_path_list,
    time_machine_view,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = {}
    
    # Create a Python file
    py_file = os.path.join(temp_dir, "test.py")
    with open(py_file, "w") as f:
        f.write("def hello():\n    print('Hello, World!')\n    return 42\n")
    files["py"] = py_file
    
    # Create a text file
    txt_file = os.path.join(temp_dir, "test.txt")
    with open(txt_file, "w") as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
    files["txt"] = txt_file
    
    # Create a large file
    large_file = os.path.join(temp_dir, "large.txt")
    with open(large_file, "w") as f:
        for i in range(100):
            f.write(f"This is line {i+1}\n")
    files["large"] = large_file
    
    # Create a subdirectory with files
    subdir = os.path.join(temp_dir, "subdir")
    os.makedirs(subdir)
    sub_file = os.path.join(subdir, "sub.txt")
    with open(sub_file, "w") as f:
        f.write("Subdirectory file content\n")
    files["sub"] = sub_file
    
    # Create a CSV file
    csv_file = os.path.join(temp_dir, "data.csv")
    with open(csv_file, "w") as f:
        f.write("name,age,city\nJohn,25,NYC\nJane,30,LA\n")
    files["csv"] = csv_file
    
    return files


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_detect_format_pdf(self):
        """Test PDF format detection."""
        assert detect_format("document.pdf") == "pdf"
        assert detect_format("Document.PDF") == "pdf"
        
    def test_detect_format_csv(self):
        """Test CSV format detection."""
        assert detect_format("data.csv") == "csv"
        
    def test_detect_format_office_docs(self):
        """Test Office document format detection."""
        assert detect_format("document.doc") == "doc"
        assert detect_format("document.docx") == "docx"
        assert detect_format("spreadsheet.xls") == "xls"
        assert detect_format("spreadsheet.xlsx") == "xlsx"
        
    def test_detect_format_unknown(self):
        """Test unknown format detection."""
        assert detect_format("unknown.xyz") == "txt"
        assert detect_format("no_extension") == "txt"
        
    def test_split_path_list_single(self):
        """Test splitting single path."""
        result = split_path_list("/path/to/file.txt")
        assert result == ["/path/to/file.txt"]
        
    def test_split_path_list_multiple(self):
        """Test splitting multiple paths."""
        result = split_path_list("/path/file1.txt, /path/file2.txt, /path/file3.txt")
        assert len(result) == 3
        assert "/path/file1.txt" in result
        assert "/path/file2.txt" in result
        assert "/path/file3.txt" in result
        
    def test_split_path_list_with_tilde(self):
        """Test path expansion with tilde."""
        with patch('src.strands_tools.file_read.expanduser') as mock_expand:
            mock_expand.side_effect = lambda x: x.replace('~', '/home/user')
            result = split_path_list("~/file1.txt, ~/file2.txt")
            assert result == ["/home/user/file1.txt", "/home/user/file2.txt"]
            
    def test_split_path_list_empty_parts(self):
        """Test handling empty parts in path list."""
        result = split_path_list("/path/file1.txt, , /path/file2.txt")
        assert len(result) == 2
        assert "/path/file1.txt" in result
        assert "/path/file2.txt" in result


class TestDocumentBlocks:
    """Test document block creation."""
    
    def test_create_document_block_success(self, sample_files):
        """Test successful document block creation."""
        doc_block = create_document_block(sample_files["txt"])
        
        assert "name" in doc_block
        assert "format" in doc_block
        assert "source" in doc_block
        assert "bytes" in doc_block["source"]
        assert doc_block["format"] == "txt"
        
    def test_create_document_block_with_format(self, sample_files):
        """Test document block creation with specified format."""
        doc_block = create_document_block(sample_files["txt"], format="csv")
        assert doc_block["format"] == "csv"
        
    def test_create_document_block_with_neutral_name(self, sample_files):
        """Test document block creation with neutral name."""
        doc_block = create_document_block(sample_files["txt"], neutral_name="custom-name")
        assert doc_block["name"] == "custom-name"
        
    def test_create_document_block_auto_name(self, sample_files):
        """Test document block creation with auto-generated name."""
        doc_block = create_document_block(sample_files["txt"])
        assert "test-" in doc_block["name"]  # Should contain base name and UUID
        
    def test_create_document_block_nonexistent_file(self):
        """Test document block creation with non-existent file."""
        with pytest.raises(Exception) as exc_info:
            create_document_block("/nonexistent/file.txt")
        assert "Error creating document block" in str(exc_info.value)
        
    def test_create_document_response(self):
        """Test document response creation."""
        documents = [{"name": "doc1", "format": "txt"}, {"name": "doc2", "format": "pdf"}]
        response = create_document_response(documents)
        
        assert response["type"] == "documents"
        assert response["documents"] == documents


class TestFileFinding:
    """Test file finding functionality."""
    
    def test_find_files_direct_file(self, sample_files):
        """Test finding a direct file path."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            files = find_files(mock_console.return_value, sample_files["txt"])
            assert files == [sample_files["txt"]]
            
    def test_find_files_directory_recursive(self, temp_dir, sample_files):
        """Test finding files in directory recursively."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            files = find_files(mock_console.return_value, temp_dir, recursive=True)
            assert len(files) >= 4  # Should find all files including subdirectory
            
    def test_find_files_directory_non_recursive(self, temp_dir, sample_files):
        """Test finding files in directory non-recursively."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            files = find_files(mock_console.return_value, temp_dir, recursive=False)
            # Should not include subdirectory files
            assert sample_files["sub"] not in files
            
    def test_find_files_glob_pattern(self, temp_dir, sample_files):
        """Test finding files with glob pattern."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            pattern = os.path.join(temp_dir, "*.txt")
            files = find_files(mock_console.return_value, pattern)
            assert len(files) >= 2  # Should find txt files
            
    def test_find_files_nonexistent_path(self):
        """Test finding files with non-existent path."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            files = find_files(mock_console.return_value, "/nonexistent/path")
            assert files == []
            
    def test_find_files_glob_error(self):
        """Test handling glob errors."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with patch('glob.glob', side_effect=Exception("Glob error")):
                files = find_files(mock_console.return_value, "*.txt")
                assert files == []


class TestRichPanel:
    """Test rich panel creation."""
    
    def test_create_rich_panel_with_file_path(self, sample_files):
        """Test creating rich panel with file path for syntax highlighting."""
        panel = create_rich_panel("def test(): pass", "Test Panel", sample_files["py"])
        assert panel.title == "[bold green]Test Panel"
        
    def test_create_rich_panel_without_file_path(self):
        """Test creating rich panel without file path."""
        panel = create_rich_panel("Plain text content", "Test Panel")
        assert panel.title == "[bold green]Test Panel"
        
    def test_create_rich_panel_no_title(self):
        """Test creating rich panel without title."""
        panel = create_rich_panel("Content")
        assert panel.title is None


class TestFileStats:
    """Test file statistics functionality."""
    
    def test_get_file_stats_success(self, sample_files):
        """Test successful file stats retrieval."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            stats = get_file_stats(mock_console.return_value, sample_files["txt"])
            
            assert "size_bytes" in stats
            assert "line_count" in stats
            assert "size_human" in stats
            assert "preview" in stats
            assert stats["line_count"] == 5  # 5 lines in test file
            
    def test_get_file_stats_large_file(self, sample_files):
        """Test file stats for large file with preview truncation."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            stats = get_file_stats(mock_console.return_value, sample_files["large"])
            
            assert stats["line_count"] == 100
            # Preview should be truncated to first 50 lines
            preview_lines = stats["preview"].split("\n")
            assert len([line for line in preview_lines if line.strip()]) <= 50


class TestFileLines:
    """Test file line reading functionality."""
    
    def test_read_file_lines_success(self, sample_files):
        """Test successful line reading."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            lines = read_file_lines(mock_console.return_value, sample_files["txt"], 1, 3)
            assert len(lines) == 2  # Lines 1-2 (0-based indexing)
            assert "Line 2" in lines[0]
            assert "Line 3" in lines[1]
            
    def test_read_file_lines_start_only(self, sample_files):
        """Test reading lines from start position only."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            lines = read_file_lines(mock_console.return_value, sample_files["txt"], 2)
            assert len(lines) == 3  # Lines 2-4 (remaining lines)
            
    def test_read_file_lines_invalid_range(self, sample_files):
        """Test reading lines with invalid range."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(ValueError) as exc_info:
                read_file_lines(mock_console.return_value, sample_files["txt"], 3, 1)
            assert "cannot be less than" in str(exc_info.value)
            
    def test_read_file_lines_nonexistent_file(self):
        """Test reading lines from non-existent file."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(FileNotFoundError):
                read_file_lines(mock_console.return_value, "/nonexistent/file.txt")
                
    def test_read_file_lines_directory(self, temp_dir):
        """Test reading lines from directory path."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(ValueError) as exc_info:
                read_file_lines(mock_console.return_value, temp_dir)
            assert "not a file" in str(exc_info.value)
            
    def test_read_file_lines_negative_start(self, sample_files):
        """Test reading lines with negative start line."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            lines = read_file_lines(mock_console.return_value, sample_files["txt"], -5, 2)
            assert len(lines) == 2  # Should start from 0


class TestFileChunk:
    """Test file chunk reading functionality."""
    
    def test_read_file_chunk_success(self, sample_files):
        """Test successful chunk reading."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            content = read_file_chunk(mock_console.return_value, sample_files["txt"], 10, 0)
            assert len(content) <= 10
            assert "Line 1" in content
            
    def test_read_file_chunk_with_offset(self, sample_files):
        """Test chunk reading with offset."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            content = read_file_chunk(mock_console.return_value, sample_files["txt"], 5, 5)
            assert len(content) <= 5
            
    def test_read_file_chunk_invalid_offset(self, sample_files):
        """Test chunk reading with invalid offset."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(ValueError) as exc_info:
                read_file_chunk(mock_console.return_value, sample_files["txt"], 10, 1000)
            assert "Invalid chunk_offset" in str(exc_info.value)
            
    def test_read_file_chunk_negative_size(self, sample_files):
        """Test chunk reading with negative size."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(ValueError) as exc_info:
                read_file_chunk(mock_console.return_value, sample_files["txt"], -10, 0)
            assert "Invalid chunk_size" in str(exc_info.value)
            
    def test_read_file_chunk_nonexistent_file(self):
        """Test chunk reading from non-existent file."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(FileNotFoundError):
                read_file_chunk(mock_console.return_value, "/nonexistent/file.txt", 10)


class TestFileSearch:
    """Test file search functionality."""
    
    def test_search_file_success(self, sample_files):
        """Test successful file search."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            results = search_file(mock_console.return_value, sample_files["txt"], "Line 2", 1)
            assert len(results) == 1
            assert results[0]["line_number"] == 2
            assert "Line 2" in results[0]["context"]
            
    def test_search_file_multiple_matches(self, sample_files):
        """Test search with multiple matches."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            results = search_file(mock_console.return_value, sample_files["txt"], "Line", 0)
            assert len(results) == 5  # All lines contain "Line"
            
    def test_search_file_no_matches(self, sample_files):
        """Test search with no matches."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            results = search_file(mock_console.return_value, sample_files["txt"], "NotFound", 1)
            assert len(results) == 0
            
    def test_search_file_case_insensitive(self, sample_files):
        """Test case-insensitive search."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            results = search_file(mock_console.return_value, sample_files["txt"], "line 2", 1)
            assert len(results) == 1  # Should find "Line 2"
            
    def test_search_file_empty_pattern(self, sample_files):
        """Test search with empty pattern."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(ValueError) as exc_info:
                search_file(mock_console.return_value, sample_files["txt"], "", 1)
            assert "cannot be empty" in str(exc_info.value)
            
    def test_search_file_nonexistent_file(self):
        """Test search in non-existent file."""
        with patch('src.strands_tools.file_read.console_util.create') as mock_console:
            mock_console.return_value = MagicMock()
            with pytest.raises(FileNotFoundError):
                search_file(mock_console.return_value, "/nonexistent/file.txt", "pattern", 1)


class TestDiffFunctionality:
    """Test diff functionality."""
    
    def test_create_diff_files(self, temp_dir):
        """Test creating diff between two files."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        
        with open(file1, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")
        with open(file2, "w") as f:
            f.write("Line 1\nModified Line 2\nLine 3\nLine 4\n")
            
        diff = create_diff(file1, file2)
        assert "Modified Line 2" in diff
        assert "Line 4" in diff
        
    def test_create_diff_identical_files(self, temp_dir):
        """Test diff between identical files."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        
        content = "Same content\n"
        with open(file1, "w") as f:
            f.write(content)
        with open(file2, "w") as f:
            f.write(content)
            
        diff = create_diff(file1, file2)
        assert diff.strip() == ""  # No differences
        
    def test_create_diff_directories(self, temp_dir):
        """Test creating diff between directories."""
        dir1 = os.path.join(temp_dir, "dir1")
        dir2 = os.path.join(temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        
        # Create files in directories
        with open(os.path.join(dir1, "common.txt"), "w") as f:
            f.write("Original content\n")
        with open(os.path.join(dir2, "common.txt"), "w") as f:
            f.write("Modified content\n")
        with open(os.path.join(dir1, "only_in_dir1.txt"), "w") as f:
            f.write("Only in dir1\n")
        with open(os.path.join(dir2, "only_in_dir2.txt"), "w") as f:
            f.write("Only in dir2\n")
            
        diff = create_diff(dir1, dir2)
        assert "common.txt" in diff
        assert "only_in_dir1.txt" in diff
        assert "only_in_dir2.txt" in diff
        
    def test_create_diff_mixed_types(self, temp_dir):
        """Test diff between file and directory (should fail)."""
        file1 = os.path.join(temp_dir, "file.txt")
        dir1 = os.path.join(temp_dir, "dir")
        
        with open(file1, "w") as f:
            f.write("Content\n")
        os.makedirs(dir1)
        
        with pytest.raises(Exception) as exc_info:
            create_diff(file1, dir1)
        assert "must be either files or directories" in str(exc_info.value)


class TestTimeMachine:
    """Test time machine functionality."""
    
    def test_time_machine_view_filesystem(self, sample_files):
        """Test time machine view with filesystem metadata."""
        result = time_machine_view(sample_files["txt"], use_git=False)
        assert "File Information" in result
        assert "Created:" in result
        assert "Modified:" in result
        assert "Size:" in result
        
    def test_time_machine_view_git_not_available(self, sample_files):
        """Test time machine view when git is not available."""
        with patch('subprocess.check_output', side_effect=Exception("Git not found")):
            with pytest.raises(Exception) as exc_info:
                time_machine_view(sample_files["txt"], use_git=True)
            assert "Error in time machine view" in str(exc_info.value)
            
    def test_time_machine_view_not_git_repo(self, sample_files):
        """Test time machine view when file is not in git repo."""
        import subprocess
        with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'git')):
            with pytest.raises(Exception) as exc_info:
                time_machine_view(sample_files["txt"], use_git=True)
            assert "not in a git repository" in str(exc_info.value)
            
    @patch('subprocess.check_output')
    def test_time_machine_view_git_success(self, mock_subprocess, sample_files):
        """Test successful git time machine view."""
        # Mock git commands
        mock_subprocess.side_effect = [
            "/repo/root\n",  # git rev-parse --show-toplevel
            "abc123|Author|2 days ago|Initial commit\ndef456|Author|1 day ago|Update file\n",  # git log
            "diff content\n",  # git blame (not used but called)
            "diff content\n",  # git show (first call)
            "diff content 2\n",  # git show (second call)
        ]
        
        result = time_machine_view(sample_files["txt"], use_git=True, num_revisions=2)
        assert "Time Machine View" in result
        assert "Git History:" in result
        assert "abc123" in result
        assert "def456" in result


class TestFileReadTool:
    """Test the main file_read tool function."""
    
    def test_file_read_missing_path(self):
        """Test file_read with missing path parameter."""
        tool = {"toolUseId": "test", "input": {"mode": "view"}}
        result = file_read(tool)
        assert result["status"] == "error"
        assert "path parameter is required" in result["content"][0]["text"]
        
    def test_file_read_missing_mode(self):
        """Test file_read with missing mode parameter."""
        tool = {"toolUseId": "test", "input": {"path": "/some/path"}}
        result = file_read(tool)
        assert result["status"] == "error"
        assert "mode parameter is required" in result["content"][0]["text"]
        
    def test_file_read_no_files_found(self):
        """Test file_read when no files are found."""
        tool = {"toolUseId": "test", "input": {"path": "/nonexistent/*", "mode": "view"}}
        result = file_read(tool)
        assert result["status"] == "error"
        assert "No files found" in result["content"][0]["text"]
        
    def test_file_read_view_mode(self, sample_files):
        """Test file_read in view mode."""
        tool = {"toolUseId": "test", "input": {"path": sample_files["txt"], "mode": "view"}}
        result = file_read(tool)
        assert result["status"] == "success"
        assert len(result["content"]) > 0
        assert "Line 1" in result["content"][0]["text"]
        
    def test_file_read_find_mode(self, temp_dir, sample_files):
        """Test file_read in find mode."""
        pattern = os.path.join(temp_dir, "*.txt")
        tool = {"toolUseId": "test", "input": {"path": pattern, "mode": "find"}}
        result = file_read(tool)
        assert result["status"] == "success"
        assert "Found" in result["content"][0]["text"]
        
    def test_file_read_lines_mode(self, sample_files):
        """Test file_read in lines mode."""
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": sample_files["txt"], 
                "mode": "lines", 
                "start_line": 1, 
                "end_line": 3
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        assert "Line 2" in result["content"][0]["text"]
        
    def test_file_read_chunk_mode(self, sample_files):
        """Test file_read in chunk mode."""
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": sample_files["txt"], 
                "mode": "chunk", 
                "chunk_size": 10, 
                "chunk_offset": 0
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        assert len(result["content"]) > 0
        
    def test_file_read_search_mode(self, sample_files):
        """Test file_read in search mode."""
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": sample_files["txt"], 
                "mode": "search", 
                "search_pattern": "Line 2", 
                "context_lines": 1
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        
    def test_file_read_stats_mode(self, sample_files):
        """Test file_read in stats mode."""
        tool = {"toolUseId": "test", "input": {"path": sample_files["txt"], "mode": "stats"}}
        result = file_read(tool)
        assert result["status"] == "success"
        stats = json.loads(result["content"][0]["text"])
        assert "size_bytes" in stats
        assert "line_count" in stats
        
    def test_file_read_preview_mode(self, sample_files):
        """Test file_read in preview mode."""
        tool = {"toolUseId": "test", "input": {"path": sample_files["txt"], "mode": "preview"}}
        result = file_read(tool)
        assert result["status"] == "success"
        assert "Line 1" in result["content"][0]["text"]
        
    def test_file_read_diff_mode(self, temp_dir):
        """Test file_read in diff mode."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        
        with open(file1, "w") as f:
            f.write("Original content\n")
        with open(file2, "w") as f:
            f.write("Modified content\n")
            
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": file1, 
                "mode": "diff", 
                "comparison_path": file2
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        
    def test_file_read_diff_mode_missing_comparison(self, sample_files):
        """Test file_read in diff mode without comparison path."""
        tool = {"toolUseId": "test", "input": {"path": sample_files["txt"], "mode": "diff"}}
        result = file_read(tool)
        assert result["status"] == "success"
        # Should have error in content about missing comparison_path
        assert any("comparison_path is required" in content.get("text", "") for content in result["content"])
        
    def test_file_read_time_machine_mode(self, sample_files):
        """Test file_read in time_machine mode."""
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": sample_files["txt"], 
                "mode": "time_machine", 
                "git_history": False
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        
    def test_file_read_document_mode(self, sample_files):
        """Test file_read in document mode."""
        tool = {"toolUseId": "test", "input": {"path": sample_files["csv"], "mode": "document"}}
        result = file_read(tool)
        assert result["status"] == "success"
        assert "document" in result["content"][0]
        
    def test_file_read_document_mode_with_format(self, sample_files):
        """Test file_read in document mode with specified format."""
        tool = {
            "toolUseId": "test", 
            "input": {
                "path": sample_files["txt"], 
                "mode": "document", 
                "format": "txt", 
                "neutral_name": "test-doc"
            }
        }
        result = file_read(tool)
        assert result["status"] == "success"
        
    def test_file_read_document_mode_error(self, temp_dir):
        """Test file_read in document mode with error."""
        nonexistent = os.path.join(temp_dir, "nonexistent.txt")
        tool = {"toolUseId": "test", "input": {"path": nonexistent, "mode": "document"}}
        result = file_read(tool)
        assert result["status"] == "error"
        assert "No files found" in result["content"][0]["text"]
        
    def test_file_read_multiple_files(self, sample_files):
        """Test file_read with multiple files."""
        paths = f"{sample_files['txt']},{sample_files['py']}"
        tool = {"toolUseId": "test", "input": {"path": paths, "mode": "view"}}
        result = file_read(tool)
        assert result["status"] == "success"
        assert len(result["content"]) >= 2  # Should have content from both files
        
    def test_file_read_file_processing_error(self, temp_dir):
        """Test file_read when file processing fails."""
        # Create a file and then make it unreadable
        test_file = os.path.join(temp_dir, "unreadable.txt")
        with open(test_file, "w") as f:
            f.write("content")
            
        # Mock file reading to fail
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            tool = {"toolUseId": "test", "input": {"path": test_file, "mode": "view"}}
            result = file_read(tool)
            assert result["status"] == "success"  # Tool succeeds but individual file fails
            assert any("Permission denied" in content.get("text", "") for content in result["content"])
            
    def test_file_read_environment_variables(self, sample_files):
        """Test file_read with environment variable defaults."""
        with patch.dict(os.environ, {
            "FILE_READ_CONTEXT_LINES_DEFAULT": "5",
            "FILE_READ_START_LINE_DEFAULT": "1",
            "FILE_READ_CHUNK_OFFSET_DEFAULT": "5"
        }):
            tool = {
                "toolUseId": "test", 
                "input": {
                    "path": sample_files["txt"], 
                    "mode": "search", 
                    "search_pattern": "Line"
                }
            }
            result = file_read(tool)
            assert result["status"] == "success"
            
    def test_file_read_general_exception(self):
        """Test file_read with general exception."""
        with patch('src.strands_tools.file_read.split_path_list', side_effect=Exception("General error")):
            tool = {"toolUseId": "test", "input": {"path": "/some/path", "mode": "view"}}
            result = file_read(tool)
            assert result["status"] == "error"
            assert "General error" in result["content"][0]["text"]