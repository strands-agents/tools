"""
File information utilities for tracking file state.

Provides utilities to extract and format file metadata for state tracking.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any


class FileInfo(TypedDict, total=False):
    """File information structure for state tracking.

    Stores file metadata
    and operation context for state tracking.
    """

    path: str
    is_dir: bool
    size: int
    created_at: str
    modified_at: str
    mode: Optional[str]
    mode_info: Optional[Dict[str, Any]]


def get_file_info(
    file_path: str,
    mode: Optional[str] = None,
    mode_info: Optional[Dict[str, Any]] = None,
) -> FileInfo:
    """
    Extract file metadata and operation context for state tracking.

    Args:
        file_path: Path to the file
        mode: Operation mode (e.g., 'view', 'write', 'find', 'search')
        mode_info: Mode-specific parameters (e.g., {'chunk_size': 1024, 'chunk_offset': 0})

    Returns:
        FileInfo dictionary with path, is_dir, size, timestamps, and operation context

    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If unable to access file metadata
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = path.stat()

    return {
        "path": str(path),
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "mode": mode,
        "mode_info": mode_info,
    }


def format_file_info(info: FileInfo) -> str:
    """
    Format file info as human-readable string.

    Args:
        info: FileInfo dictionary

    Returns:
        Formatted string representation
    """
    file_type = "directory" if info.get("is_dir") else "file"
    size_kb = info.get("size", 0) / 1024

    lines = [
        f"Path: {info.get('path')}",
        f"Type: {file_type}",
        f"Size: {size_kb:.2f} KB",
        f"Created: {info.get('created_at')}",
        f"Modified: {info.get('modified_at')}",
    ]

    if info.get("mode"):
        lines.append(f"Mode: {info['mode']}")

    if info.get("mode_info"):
        mode_info_str = ", ".join(f"{k}={v}" for k, v in info["mode_info"].items())
        lines.append(f"Mode Info: {mode_info_str}")

    return "\n".join(lines)
