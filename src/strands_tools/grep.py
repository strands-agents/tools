"""
grep tool for fast content search across files using ripgrep.

This module provides a lightweight file discovery tool that searches file contents
using regular expressions and returns file paths containing matches. Built on ripgrep
for exceptional performance on large codebases.

Features:
- Fast regex-based content search powered by ripgrep
- Returns file paths only (not content) for efficient discovery workflows
- Supports file filtering via glob patterns (e.g., "*.py", "*.{ts,tsx}")
- Results sorted by modification time (most recent first)
- 30-second timeout for safety

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import grep

agent = Agent(tools=[grep])

# Find all files containing pattern
agent.tool.grep(pattern="TODO")

# Search specific directory with file filter
agent.tool.grep(pattern="function\\s+\\w+", path="src/", include="*.py")

# Find deprecated API usage
agent.tool.grep(pattern="old_api_function", include="*.{js,ts}")
```

Requirements:
- ripgrep (rg) must be installed on the system:
  - macOS: brew install ripgrep
  - Ubuntu: apt install ripgrep
  - Windows: choco install ripgrep
"""

import subprocess
from pathlib import Path
from typing import Optional

from strands import tool


@tool
def grep(
    pattern: str,
    path: Optional[str] = None,
    include: Optional[str] = None,
) -> str:
    """
    Fast content search tool across files.

    - Searches file contents using regular expressions
    - Supports full regex syntax (e.g. "log.*Error", "function\\s+\\w+", etc.)
    - Filter files by pattern with the include parameter (e.g. "*.js", "*.{ts,tsx}")
    - Returns file paths with at least one match sorted by modification time
    - Use this tool when you need to find files containing specific patterns

    Args:
        pattern: The regular expression pattern to search for in file contents
        path: The directory to search in. Defaults to current working directory.
        include: File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")

    Returns:
        File paths with at least one match, sorted by modification time
    """
    # Build ripgrep command
    cmd = ["rg", "-l", pattern]  # -l = files-with-matches (paths only)

    search_path = path or "."
    cmd.append(search_path)

    if include:
        cmd.extend(["--glob", include])

    try:
        # Execute ripgrep
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=Path.cwd())

        # Handle results
        # returncode 0 = matches found, 1 = no matches, 2+ = error
        if result.returncode == 0:
            # Parse paths and convert to absolute
            file_paths = [str(Path(line.strip()).resolve()) for line in result.stdout.splitlines() if line.strip()]

            # Sort by modification time (most recent first)
            try:
                file_paths = sorted(file_paths, key=lambda p: Path(p).stat().st_mtime, reverse=True)
            except Exception:
                pass  # Keep original order if sorting fails

            return "\n".join(file_paths)

        elif result.returncode == 1:
            return f"No files found matching pattern: {pattern}"

        else:
            # Error occurred
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error searching: {error_msg}"

    except subprocess.TimeoutExpired:
        return "Search timed out (30s limit)"

    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep."

    except Exception as e:
        return f"Error: {str(e)}"
