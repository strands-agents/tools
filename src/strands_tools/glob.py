"""
glob tool for file pattern matching using standard glob syntax.

This module provides a simple, efficient tool for discovering files by name patterns.
Uses Python's built-in pathlib for zero external dependencies and cross-platform
compatibility.

Features:
- Standard glob syntax support including ** for recursive matching
- Returns file paths only (excludes directories)
- Results sorted by modification time (most recent first)
- Zero external dependencies (uses pathlib from standard library)
- Cross-platform compatibility

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import glob

agent = Agent(tools=[glob])

# Find all JSON files recursively
agent.tool.glob(pattern="**/*.json")

# Find Python files in specific directory
agent.tool.glob(pattern="**/*.py", path="src/")

# Find test files
agent.tool.glob(pattern="test_*.py", path="tests/")
```

Glob Pattern Syntax:
- * matches any characters except /
- ** matches any characters including /
- ? matches a single character
- [abc] matches one character from the set
"""

from pathlib import Path
from typing import Optional

from strands import tool


@tool
def glob(
    pattern: str,
    path: Optional[str] = None,
) -> str:
    """
    Fast file pattern matching tool across files.

    - Supports glob patterns like "**/*.js" or "src/**/*.ts"
    - Returns matching file paths sorted by modification time
    - Use this tool when you need to find files by name patterns
    - You can call multiple tools in a single response. It is always better to speculatively
      perform multiple searches in parallel if they are potentially useful.

    Args:
        pattern: The glob pattern to match files against
        path: The directory to search in. If not specified, the current working directory will be used.

    Returns:
        Matching file paths sorted by modification time
    """
    search_path = Path(path) if path else Path.cwd()

    try:
        # Use pathlib glob
        matches = list(search_path.glob(pattern))

        # Filter to files only (exclude directories)
        file_paths = [str(p.resolve()) for p in matches if p.is_file()]

        if not file_paths:
            return f"No files found matching pattern: {pattern}"

        # Sort by modification time (most recent first)
        try:
            file_paths = sorted(file_paths, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        except Exception:
            pass  # Keep original order if sorting fails

        return "\n".join(file_paths)

    except Exception as e:
        return f"Error: {str(e)}"
