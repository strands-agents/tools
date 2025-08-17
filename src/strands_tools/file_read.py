"""
Advanced file reading tool for Strands Agent with multifaceted capabilities.

This module provides a comprehensive file reading capability with rich output formatting,
pattern searching, document mode support, and multiple specialized reading modes. It's designed
to handle various file reading scenarios, from simple content viewing to complex operations
like diffs and version history analysis.

Key Features:

1. Multiple Reading Modes:
   • view: Display full file contents with syntax highlighting
   • find: List matching files with directory tree visualization
   • lines: Show specific line ranges with context
   • chunk: Read byte chunks from specific offsets
   • search: Pattern searching with context highlighting
   • stats: File statistics and metrics
   • preview: Quick content preview
   • diff: Compare files or directories
   • time_machine: View version history
   • document: Generate Bedrock document blocks

2. Rich Output Display:
   • Syntax highlighting based on file type
   • Formatted panels for better readability
   • Directory tree visualization
   • Line numbering and statistics
   • Beautiful console output with panels and tables

3. Advanced Capabilities:
   • Multi-file support with comma-separated paths
   • Wildcard pattern matching
   • Recursive directory traversal
   • Git integration for version history
   • Document format detection
   • Bedrock document block generation

4. Context-Aware Features:
   • Smart line finding with context
   • Highlighted search results
   • Diff visualization
   • File metadata extraction
   • Version control integration

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import file_read

agent = Agent(tools=[file_read])

# View file content with syntax highlighting
agent.tool.file_read(path="/path/to/file.py", mode="view")

# List files matching a pattern
agent.tool.file_read(path="/path/to/project/*.py", mode="find")

# Read specific line ranges
agent.tool.file_read(
    path="/path/to/file.txt",
    mode="lines",
    start_line=10,
    end_line=20
)

# Search for patterns
agent.tool.file_read(
    path="/path/to/file.txt",
    mode="search",
    search_pattern="function",
    context_lines=3
)

# Compare files
agent.tool.file_read(
    path="/path/to/file1.txt",
    mode="diff",
    comparison_path="/path/to/file2.txt"
)

# View file history
agent.tool.file_read(
    path="/path/to/file.py",
    mode="time_machine",
    git_history=True,
    num_revisions=5
)

# Generate document blocks for Bedrock
agent.tool.file_read(
    path="/path/to/document.pdf",
    mode="document"
)
```

See the file_read function docstring for more details on modes and parameters.
"""

import glob
import json
import os
import time as time_module
import uuid
from os.path import expanduser
from typing import Any, Dict, List, Optional, Union, cast

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from strands.types.media import DocumentContent
from strands.types.tools import (
    ToolResult,
    ToolResultContent,
    ToolUse,
)

from strands_tools.utils import console_util
from strands_tools.utils.detect_language import detect_language

# Document format mapping
FORMAT_EXTENSIONS = {
    "pdf": [".pdf"],
    "csv": [".csv"],
    "doc": [".doc"],
    "docx": [".docx"],
    "xls": [".xls"],
    "xlsx": [".xlsx"],
    # Given extensions below can be added as document block but document blocks are limited to 5 in every conversation.
    # "html": [".html", ".htm"],
    # "txt": [".txt"],
    # "md": [".md", ".markdown"]
}

# Reverse mapping for format detection
EXTENSION_TO_FORMAT = {ext: fmt for fmt, exts in FORMAT_EXTENSIONS.items() for ext in exts}


def detect_format(file_path: str) -> str:
    """
    Detect document format from file extension.

    Examines the file extension to determine the appropriate document format
    for Bedrock compatibility in document mode.

    Args:
        file_path: Path to the file

    Returns:
        str: Detected format identifier or 'txt' as fallback
    """
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_TO_FORMAT.get(ext, "txt")


def create_document_block(
    file_path: str, format: Optional[str] = None, neutral_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a Bedrock document block from a file.

    Reads the file content, encodes it appropriately, and creates a document block
    structure suitable for use with Bedrock document processing capabilities.

    Args:
        file_path: Path to the file
        format: Optional document format. If None, detected from extension.
        neutral_name: Optional neutral document name. If None, generated from filename.

    Returns:
        Dict[str, Any]: Document block structure ready for Bedrock

    Raises:
        Exception: If there is an error reading or encoding the file
    """
    try:
        # Detect format if not provided
        if not format:
            format = detect_format(file_path)

        # Create neutral name if not provided
        if not neutral_name:
            base_name = os.path.basename(file_path)
            name_uuid = str(uuid.uuid4())[:8]
            neutral_name = f"{os.path.splitext(base_name)[0]}-{name_uuid}"

        # Read file content
        with open(file_path, "rb") as f:
            content = f.read()

        # Create document block
        return {"name": neutral_name, "format": format, "source": {"bytes": content}}

    except Exception as e:
        raise Exception(f"Error creating document block for {file_path}: {str(e)}") from e


def create_document_response(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a response containing document blocks.

    Formats a list of document blocks into the proper response structure
    for Bedrock document processing.

    Args:
        documents: List of document blocks created by create_document_block()

    Returns:
        Dict[str, Any]: Response structure with document blocks
    """
    return {"type": "documents", "documents": documents}


def split_path_list(path: str) -> List[str]:
    """
    Split comma-separated path list and expand each path.

    Handles multiple file paths provided as comma-separated values,
    expanding user paths (e.g., ~/) in each one.

    Args:
        path: Comma-separated list of file paths

    Returns:
        List[str]: List of expanded paths
    """
    paths = [p.strip() for p in path.split(",") if p.strip()]
    return [expanduser(p) for p in paths]


TOOL_SPEC = {
    "name": "file_read",
    "description": (
        "File reading tool with search capabilities, various reading modes, and document mode support "
        "for Bedrock compatibility.\n\n"
        "Features:\n"
        "1. Multi-file support (comma-separated paths)\n"
        "2. Full document format support (pdf, doc, docx, etc.)\n"
        "3. Search and filtering capabilities\n"
        "4. Version control integration\n"
        "5. Document block generation for Bedrock\n\n"
        "Modes:\n"
        "- find: List matching files\n"
        "- view: Display file contents\n"
        "- lines: Show specific line ranges\n"
        "- chunk: Read byte chunks\n"
        "- search: Pattern searching\n"
        "- stats: File statistics\n"
        "- preview: Quick content preview\n"
        "- diff: Compare files/directories\n"
        "- time_machine: Version history\n"
        "- document: Generate Bedrock document blocks"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path(s) to file(s). For multiple files, use comma-separated list: "
                        "'file1.txt,file2.md,data/*.json'"
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "Reading mode: find, view, lines, chunk, search, stats, preview, diff, time_machine, document"
                    ),
                    "enum": [
                        "find",
                        "view",
                        "lines",
                        "chunk",
                        "search",
                        "stats",
                        "preview",
                        "diff",
                        "time_machine",
                        "document",
                    ],
                },
                "format": {
                    "type": "string",
                    "description": "Document format for document mode (autodetected if not specified)",
                    "enum": [
                        "pdf",
                        "csv",
                        "doc",
                        "docx",
                        "xls",
                        "xlsx",
                        "html",
                        "txt",
                        "md",
                    ],
                },
                "neutral_name": {
                    "type": "string",
                    "description": "Neutral document name to prevent prompt injection (default: filename-UUID)",
                },
                "comparison_path": {
                    "type": "string",
                    "description": "Second file/directory path for diff mode comparison",
                },
                "diff_type": {
                    "type": "string",
                    "description": "Type of diff view (unified diff)",
                    "enum": ["unified"],
                    "default": "unified",
                },
                "git_history": {
                    "type": "boolean",
                    "description": "Whether to use git history for time_machine mode",
                    "default": True,
                },
                "num_revisions": {
                    "type": "integer",
                    "description": "Number of revisions to show in time_machine mode",
                    "default": 5,
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (for lines mode)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (for lines mode)",
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of chunk in bytes (for chunk mode)",
                },
                "chunk_offset": {
                    "type": "integer",
                    "description": "Offset in bytes (for chunk mode)",
                },
                "search_pattern": {
                    "type": "string",
                    "description": "Pattern to search for (for search mode)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines around search results",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories (default: true)",
                    "default": True,
                },
            },
            "required": ["path", "mode"],
        }
    },
}


def find_files(console: Console, pattern: str, recursive: bool = True) -> List[str]:
    """
    Find files matching the pattern with better error handling.

    Supports glob patterns, direct file paths, and directory traversal
    with configurable recursion for finding matching files.

    Args:
        pattern: File pattern to match (can include wildcards)
        recursive: Whether to search recursively through subdirectories

    Returns:
        List[str]: List of matching file paths
    """
    try:
        # Consistent path normalization
        pattern = expanduser(pattern)

        # Direct file/directory check first
        if os.path.exists(pattern):
            if os.path.isfile(pattern):
                return [pattern]
            elif os.path.isdir(pattern):
                matching_files = []

                for root, _dirs, files in os.walk(pattern):
                    if not recursive and root != pattern:
                        continue

                    for file in sorted(files):
                        if not file.startswith("."):  # Skip hidden files
                            matching_files.append(os.path.join(root, file))

                return sorted(matching_files)

        # Handle glob patterns
        if recursive and "**" not in pattern:
            # Add recursive glob pattern
            base_dir = os.path.dirname(pattern)
            file_pattern = os.path.basename(pattern)
            pattern = os.path.join(base_dir if base_dir else ".", "**", file_pattern)

        try:
            matching_files = glob.glob(pattern, recursive=recursive)
            return sorted(matching_files)
        except Exception as e:
            console.print(
                Panel(
                    escape(f"Warning: Error while globbing {pattern}: {e}"),
                    title="[yellow]Warning",
                    border_style="yellow",
                )
            )
            return []

    except Exception as e:
        console.print(Panel(escape(f"Error in find_files: {str(e)}"), title="[red]Error", border_style="red"))
        return []


def create_rich_panel(content: str, title: Optional[str] = None, file_path: Optional[str] = None) -> Panel:
    """
    Create a Rich panel with optional syntax highlighting.

    Generates a visually appealing panel containing the provided content,
    with optional syntax highlighting based on the file type if a file path is provided.

    Args:
        content: Content to display in panel
        title: Optional panel title
        file_path: Optional path to file for language detection and syntax highlighting

    Returns:
        Panel: Rich panel object for console display
    """
    if file_path:
        language = detect_language(file_path)
        syntax = Syntax(content, language, theme="monokai", line_numbers=True)
        content_for_panel: Union[Syntax, Text] = syntax
    else:
        content_for_panel = Text(content)

    return Panel(
        content_for_panel,
        title=f"[bold green]{title}" if title else None,
        border_style="blue",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 2),
    )


def get_file_stats(console, file_path: str) -> Dict[str, Any]:
    """
    Get file statistics including size, line count, and preview.

    Analyzes a file to gather key metrics like size and line count,
    and generates a preview of the first 50 lines.

    Args:
        file_path: Path to the file

    Returns:
        Dict[str, Any]: File statistics including size_bytes, line_count,
                        size_human (formatted size), and preview
    """
    file_path = expanduser(file_path)
    stats: Dict[str, Any] = {
        "size_bytes": os.path.getsize(file_path),
        "line_count": 0,
        "preview": "",
    }

    with open(file_path, "r") as f:
        preview_lines = []
        for i, line in enumerate(f):
            stats["line_count"] += 1
            if i < 50:  # First 50 lines as preview
                preview_lines.append(line)

    stats["preview"] = "\n".join(preview_lines)
    stats["size_human"] = f"{stats['size_bytes'] / 1024:.2f} KB"

    table = Table(title="File Statistics", box=box.DOUBLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("File Size", stats["size_human"])
    table.add_row("Line Count", str(stats["line_count"]))
    table.add_row("File Path", file_path)

    console.print(table)
    return stats


def read_file_lines(console: Console, file_path: str, start_line: int = 0, end_line: Optional[int] = None) -> List[str]:
    """
    Read specific lines from file.

    Extracts and returns a specific range of lines from a file,
    with validation of line range parameters.

    Args:
        file_path: Path to the file
        start_line: First line to read (0-based)
        end_line: Last line to read (optional)

    Returns:
        List[str]: List of lines read

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the path is not a file or line numbers are invalid
    """
    file_path = expanduser(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r") as f:
            all_lines = f.readlines()

        # Validate line numbers
        start_line = max(start_line, 0)

        if end_line is not None:
            end_line = min(end_line, len(all_lines))
            if end_line < start_line:
                raise ValueError(f"end_line ({end_line}) cannot be less than start_line ({start_line})")

        lines = all_lines[start_line:end_line]

        # Create a preview panel
        line_range = f"{start_line + 1}-{end_line if end_line else len(all_lines)}"
        panel = Panel(
            escape("".join(lines)),
            title=f"[bold green]Lines {line_range} from {os.path.basename(file_path)}",
            border_style="blue",
            expand=False,
        )
        console.print(panel)
        return lines

    except Exception as e:
        error_panel = Panel(escape(f"Error reading file: {str(e)}"), title="[bold red]Error", border_style="red")
        console.print(error_panel)
        raise


def read_file_chunk(console: Console, file_path: str, chunk_size: int, chunk_offset: int = 0) -> str:
    """
    Read a chunk of file from given offset.

    Reads a specific byte range from a file, starting at the specified offset
    and containing the requested number of bytes.

    Args:
        file_path: Path to the file
        chunk_size: Number of bytes to read
        chunk_offset: Starting offset in bytes

    Returns:
        str: Content read from file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the path is not a file or chunk parameters are invalid
    """
    file_path = expanduser(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        file_size = os.path.getsize(file_path)
        if chunk_offset < 0 or chunk_offset > file_size:
            raise ValueError(f"Invalid chunk_offset: {chunk_offset}. File size is {file_size} bytes.")

        if chunk_size < 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        with open(file_path, "r") as f:
            f.seek(chunk_offset)
            content = f.read(chunk_size)

        # Create information panel
        file_name = os.path.basename(file_path)
        info = (
            f"File: {file_name}\n"
            f"Total size: {file_size} bytes\n"
            f"Chunk offset: {chunk_offset} bytes\n"
            f"Chunk size: {chunk_size} bytes\n"
            f"Content length: {len(content)} bytes"
        )

        info_panel = Panel(
            info,
            title="[bold yellow]Chunk Information",
            border_style="yellow",
            expand=False,
        )
        console.print(info_panel)

        # Create content panel
        content_panel = Panel(
            escape(content),
            title=f"[bold green]Content from {file_name}",
            border_style="blue",
            expand=False,
        )
        console.print(content_panel)

        return content

    except Exception as e:
        error_panel = Panel(
            escape(f"Error reading file chunk: {str(e)}"),
            title="[bold red]Error",
            border_style="red",
        )
        console.print(error_panel)
        raise


def search_file(console: Console, file_path: str, pattern: str, context_lines: int = 2) -> List[Dict[str, Any]]:
    """
    Search file for pattern and return matches with context.

    Searches for a text pattern within a file and returns matching lines
    with the specified number of context lines before and after each match.

    Args:
        file_path: Path to the file
        pattern: Text pattern to search for
        context_lines: Number of lines of context around matches

    Returns:
        List[Dict[str, Any]]: List of matches with line number and context

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the path is not a file or pattern is empty
    """
    file_path = expanduser(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    if not pattern:
        raise ValueError("Search pattern cannot be empty")

    results = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        total_matches = 0
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                total_matches += 1
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context_text = []
                for ctx_idx in range(start, end):
                    prefix = "  "
                    if ctx_idx == i:
                        prefix = "→ "  # Highlight the matching line
                    line_text = lines[ctx_idx].rstrip()
                    # Highlight the matching pattern in the line
                    if ctx_idx == i:
                        pattern_idx = line_text.lower().find(pattern.lower())
                        if pattern_idx != -1:
                            line_text = (
                                line_text[:pattern_idx]
                                + f"[bold yellow]{line_text[pattern_idx : pattern_idx + len(pattern)]}[/bold yellow]"
                                + line_text[pattern_idx + len(pattern) :]
                            )
                    context_text.append(f"{prefix}{ctx_idx + 1}: {line_text}")

                match_text = "\n".join(context_text)
                # Create a panel for each match
                panel = Panel(
                    escape(match_text),
                    title=f"[bold green]Match at line {i + 1}",
                    border_style="blue",
                    expand=False,
                )
                console.print(panel)

                results.append({"line_number": i + 1, "context": match_text})

        # Print summary
        summary = Panel(
            escape(f"Found {total_matches} matches for pattern '{pattern}' in {os.path.basename(file_path)}"),
            title="[bold yellow]Search Summary",
            border_style="yellow",
            expand=False,
        )
        console.print(summary)

        return results

    except Exception as e:
        error_panel = Panel(
            escape(f"Error searching file: {str(e)}"),
            title="[bold red]Error",
            border_style="red",
        )
        console.print(error_panel)
        raise


def create_diff(file_path: str, comparison_path: str, diff_type: str = "unified") -> str:
    """
    Create a diff between two files or directories.

    Compares two files or directories and generates a diff output showing
    the differences between them.

    Args:
        file_path: Path to the first file/directory
        comparison_path: Path to the second file/directory
        diff_type: Type of diff view ('unified' is currently supported)

    Returns:
        str: Formatted diff output

    Raises:
        Exception: If there's an error during diff creation or paths are invalid
    """
    try:
        import difflib
        from pathlib import Path

        file_path = expanduser(file_path)
        comparison_path = expanduser(comparison_path)

        # Function to read file content
        def read_file(path: str) -> List[str]:
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines()

        # Handle directory comparison
        if os.path.isdir(file_path) and os.path.isdir(comparison_path):
            diff_results = []

            # Get all files in both directories
            def get_files(path: str) -> set:
                return set(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())

            files1 = get_files(file_path)
            files2 = get_files(comparison_path)

            # Compare files
            all_files = sorted(files1 | files2)
            for file in all_files:
                path1 = os.path.join(file_path, file)
                path2 = os.path.join(comparison_path, file)

                if file in files1 and file in files2:
                    # Both files exist - compare content
                    diff = create_diff(path1, path2, diff_type)
                    if diff.strip():  # Only include if there are differences
                        diff_results.append(f"\n=== {file} ===\n{diff}")
                elif file in files1:
                    diff_results.append(f"\n=== {file} ===\nOnly in {file_path}")
                else:
                    diff_results.append(f"\n=== {file} ===\nOnly in {comparison_path}")

            return "\n".join(diff_results)

        # Handle single file comparison
        elif os.path.isfile(file_path) and os.path.isfile(comparison_path):
            lines1 = read_file(file_path)
            lines2 = read_file(comparison_path)

            # Create unified diff
            diff_iter = difflib.unified_diff(
                lines1,
                lines2,
                fromfile=os.path.basename(file_path),
                tofile=os.path.basename(comparison_path),
                lineterm="",
            )
            return "\n".join(list(diff_iter))
        else:
            raise ValueError("Both paths must be either files or directories")

    except Exception as e:
        raise Exception(f"Error creating diff: {str(e)}") from e


def time_machine_view(file_path: str, use_git: bool = True, num_revisions: int = 5) -> str:
    """
    Show file history using git or filesystem metadata.

    Retrieves and displays the version history of a file using either
    git history (if available) or filesystem metadata.

    Args:
        file_path: Path to the file
        use_git: Whether to use git history if available
        num_revisions: Number of revisions to show

    Returns:
        str: Formatted history output

    Raises:
        Exception: If there's an error retrieving file history
    """
    try:
        file_path = os.path.expanduser(file_path)

        if use_git:
            import subprocess

            # Check if file is in a git repository
            try:
                repo_root = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=os.path.dirname(file_path),
                    stderr=subprocess.PIPE,
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                raise ValueError("File is not in a git repository") from None

            # Get relative path from repo root
            rel_path = os.path.relpath(file_path, repo_root)

            # Get git log
            log_output = subprocess.check_output(
                [
                    "git",
                    "log",
                    "-n",
                    str(num_revisions),
                    "--pretty=format:%h|%an|%ar|%s",
                    "--",
                    rel_path,
                ],
                cwd=repo_root,
                text=True,
            ).split("\n")

            # Get blame information
            subprocess.check_output(["git", "blame", "--line-porcelain", rel_path], cwd=repo_root, text=True)

            # Process git information
            history = []
            current_commit = None

            for line in log_output:
                if line:
                    commit_hash, author, time, message = line.split("|")

                    if not current_commit:
                        current_commit = commit_hash

                    # Get changes in this commit
                    try:
                        changes = subprocess.check_output(
                            [
                                "git",
                                "show",
                                "--format=",
                                "--patch",
                                commit_hash,
                                "--",
                                rel_path,
                            ],
                            cwd=repo_root,
                            text=True,
                        )
                    except subprocess.CalledProcessError:
                        changes = "Unable to retrieve changes"

                    history.append(
                        {
                            "commit": commit_hash,
                            "author": author,
                            "time": time,
                            "message": message,
                            "changes": changes,
                        }
                    )

            # Format output
            output = []
            output.append(f"=== Time Machine View for {os.path.basename(file_path)} ===\n")
            output.append("Git History:\n")

            for entry in history:
                output.append(f"Commit: {entry['commit']}")
                output.append(f"Author: {entry['author']}")
                output.append(f"Time: {entry['time']}")
                output.append(f"Message: {entry['message']}")
                output.append("\nChanges:")
                output.append(entry["changes"])
                output.append("-" * 40 + "\n")

            return "\n".join(output)

        else:
            # Fallback to filesystem metadata
            stat = os.stat(file_path)

            output = []
            output.append(f"=== File Information for {os.path.basename(file_path)} ===\n")
            output.append(f"Created: {time_module.ctime(stat.st_ctime)}")
            output.append(f"Modified: {time_module.ctime(stat.st_mtime)}")
            output.append(f"Accessed: {time_module.ctime(stat.st_atime)}")
            output.append(f"Size: {stat.st_size:,} bytes")
            output.append(f"Owner: {stat.st_uid}")
            output.append(f"Permissions: {oct(stat.st_mode)[-3:]}")

            return "\n".join(output)

    except Exception as e:
        raise Exception(f"Error in time machine view: {str(e)}") from e


def file_read(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Advanced file reading tool with multiple specialized reading modes.

    This tool provides comprehensive file reading capabilities with support for
    multiple specialized modes, from simple content viewing to complex file operations
    like diff comparisons and version history analysis. It handles multiple file paths,
    pattern matching, and can generate document blocks for Bedrock compatibility.

    How It Works:
    ------------
    1. Parses the input parameters to determine the requested mode
    2. Validates the required parameters for that mode
    3. Finds all files matching the provided path patterns
    4. Processes each file according to the requested mode
    5. Formats the results with rich output and appropriate structure
    6. Returns the results or appropriate error messages

    Reading Modes:
    ------------
    - find: Lists all files matching the pattern (supports wildcards)
    - view: Shows full file contents with syntax highlighting
    - lines: Shows specific line ranges from files
    - chunk: Reads binary chunks from files at specific offsets
    - search: Searches for patterns with context highlighting
    - stats: Displays file statistics like size and line count
    - preview: Shows a quick preview of file content
    - diff: Compares two files or directories and shows differences
    - time_machine: Shows version history from git or filesystem
    - document: Generates Bedrock document blocks for file content

    Common Usage Scenarios:
    --------------------
    - Reading code files with syntax highlighting
    - Searching for specific patterns in logs or source code
    - Comparing different versions of files or directories
    - Analyzing file metadata and statistics
    - Reading only specific parts of large files
    - Examining file version history
    - Preparing file content for Bedrock document processing

    Args:
        tool: ToolUse object containing the following input fields:
            - path: Path(s) to file(s). For multiple files, use comma-separated list.
              Can include wildcards like '*.py' or directories.
            - mode: Reading mode to use (required)
            - Additional parameters specific to each mode
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing status and response content in the format:
        {
            "toolUseId": "<tool_use_id>",
            "status": "success|error",
            "content": [{"text": "Response message"}]
        }

    Notes:
        - Document mode returns document blocks for Bedrock compatibility
        - Multiple files can be processed in a single call with comma-separated paths
        - The tool supports various wildcard patterns for matching multiple files
        - Document format is auto-detected from file extension or can be specified
        - For diff mode, both paths must be either files or directories
    """
    console = console_util.create()

    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    # Get environment variables at runtime
    file_read_recursive_default = os.getenv("FILE_READ_RECURSIVE_DEFAULT", "true").lower() == "true"
    file_read_context_lines_default = int(os.getenv("FILE_READ_CONTEXT_LINES_DEFAULT", "2"))
    file_read_start_line_default = int(os.getenv("FILE_READ_START_LINE_DEFAULT", "0"))
    file_read_chunk_offset_default = int(os.getenv("FILE_READ_CHUNK_OFFSET_DEFAULT", "0"))
    file_read_diff_type_default = os.getenv("FILE_READ_DIFF_TYPE_DEFAULT", "unified")
    file_read_use_git_default = os.getenv("FILE_READ_USE_GIT_DEFAULT", "true").lower() == "true"
    file_read_num_revisions_default = int(os.getenv("FILE_READ_NUM_REVISIONS_DEFAULT", "5"))

    try:
        # Validate required parameters
        if not tool_input.get("path"):
            raise ValueError("path parameter is required")

        if not tool_input.get("mode"):
            raise ValueError("mode parameter is required")

        # Get input parameters
        mode = tool_input["mode"]
        paths = split_path_list(tool_input["path"])  # Handle comma-separated paths
        recursive = tool_input.get("recursive", file_read_recursive_default)

        # Find all matching files across all paths
        matching_files = []
        for path_pattern in paths:
            files = find_files(console, path_pattern, recursive)
            matching_files.extend(files)

        matching_files = sorted(set(matching_files))  # Remove duplicates

        if not matching_files:
            error_msg = f"No files found matching pattern(s): {', '.join(paths)}"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_msg}],
            }

        # Special handling for document mode
        if mode == "document":
            try:
                format = tool_input.get("format")
                neutral_name = tool_input.get("neutral_name")

                # Create document blocks for each file
                document_blocks = []
                for file_path in matching_files:
                    try:
                        document_blocks.append(
                            create_document_block(file_path, format=format, neutral_name=neutral_name)
                        )
                    except Exception as e:
                        console.print(
                            Panel(
                                escape(f"Error creating document block for {file_path}: {str(e)}"),
                                title="[bold yellow]Warning",
                                border_style="yellow",
                            )
                        )

                # Create response with document blocks
                document_content: List[ToolResultContent] = []
                for doc in document_blocks:
                    document_content.append({"document": cast(DocumentContent, doc)})

                return {
                    "toolUseId": tool_use_id,
                    "status": "success",
                    "content": document_content,
                }

            except Exception as e:
                error_msg = f"Error in document mode: {str(e)}"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": error_msg}],
                }

        response_content: List[ToolResultContent] = []

        # Handle find mode
        if mode == "find":
            tree = Tree("🔍 Found Files")
            files_by_dir: Dict[str, List[str]] = {}

            # Group files by directory
            for file_path in matching_files:
                dir_path = os.path.dirname(file_path) or "."
                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []
                files_by_dir[dir_path].append(os.path.basename(file_path))

            # Create tree structure
            for dir_path, files in sorted(files_by_dir.items()):
                dir_node = tree.add(f"📁 {dir_path}")
                for file_name in sorted(files):
                    dir_node.add(f"📄 {file_name}")

            # Display results
            console.print(Panel(tree, title="[bold green]File Tree", border_style="blue"))
            console.print(
                Panel(
                    escape("\n".join(matching_files)),
                    title="[bold green]File Paths",
                    border_style="blue",
                )
            )

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Found {len(matching_files)} files:\n" + "\n".join(matching_files)}],
            }

        # Process each file for other modes
        for file_path in matching_files:
            try:
                if mode == "view":
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Create rich panel with syntax highlighting
                        view_panel = create_rich_panel(
                            content,
                            f"📄 {os.path.basename(file_path)}",
                            file_path,
                        )
                        console.print(view_panel)
                        response_content.append({"text": f"Content of {file_path}:\n{content}"})
                    except Exception as e:
                        error_msg = f"Error reading file {file_path}: {str(e)}"
                        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                        response_content.append({"text": error_msg})

                elif mode == "preview":
                    stats = get_file_stats(console, file_path)
                    with open(file_path, "r") as f:
                        content = "".join(f.readlines()[:50])

                    preview_panel = create_rich_panel(
                        content,
                        (
                            f"📄 Preview: {os.path.basename(file_path)} "
                            f"(first 50 lines of {stats['line_count']} total lines)"
                        ),
                        file_path,
                    )
                    console.print(preview_panel)
                    response_content.append(
                        {
                            "text": (
                                f"File: {file_path}\nSize: {stats['size_human']}\n"
                                f"Total Lines: {stats['line_count']}\n\nPreview:\n{content}"
                            )
                        }
                    )

                elif mode == "stats":
                    stats = get_file_stats(console, file_path)
                    response_content.append({"text": json.dumps(stats, indent=2)})

                elif mode == "lines":
                    lines = read_file_lines(
                        console,
                        file_path,
                        tool_input.get("start_line", file_read_start_line_default),
                        tool_input.get("end_line"),
                    )
                    response_content.append({"text": "".join(lines)})

                elif mode == "chunk":
                    content = read_file_chunk(
                        console,
                        file_path,
                        tool_input.get("chunk_size", 1024),
                        tool_input.get("chunk_offset", file_read_chunk_offset_default),
                    )
                    response_content.append({"text": content})

                elif mode == "search":
                    results = search_file(
                        console,
                        file_path,
                        tool_input.get("search_pattern", ""),
                        tool_input.get("context_lines", file_read_context_lines_default),
                    )
                    response_content.extend([{"text": r["context"]} for r in results])

                elif mode == "diff":
                    comparison_path = tool_input.get("comparison_path")
                    if not comparison_path:
                        raise ValueError("comparison_path is required for diff mode")

                    diff_output = create_diff(
                        file_path,
                        os.path.expanduser(comparison_path),
                        tool_input.get("diff_type", file_read_diff_type_default),
                    )

                    diff_panel = create_rich_panel(
                        diff_output,
                        f"Diff: {os.path.basename(file_path)} vs {os.path.basename(comparison_path)}",
                        file_path,
                    )
                    console.print(diff_panel)
                    response_content.append({"text": f"Diff between {file_path} and {comparison_path}:\n{diff_output}"})

                elif mode == "time_machine":
                    history_output = time_machine_view(
                        file_path,
                        tool_input.get("git_history", file_read_use_git_default),
                        tool_input.get("num_revisions", file_read_num_revisions_default),
                    )

                    history_panel = create_rich_panel(
                        history_output,
                        f"Time Machine: {os.path.basename(file_path)}",
                        file_path,
                    )
                    console.print(history_panel)
                    response_content.append({"text": f"Time Machine view for {file_path}:\n{history_output}"})

            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                response_content.append({"text": error_msg})

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": response_content,
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [cast(ToolResultContent, {"text": error_msg})],
        }
