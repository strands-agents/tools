import os


def detect_language(file_path: str) -> str:
    """Detect language for syntax highlighting based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".java": "java",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".tsx": "typescript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".php": "php",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".xml": "xml",
        ".sql": "sql",
        ".r": "r",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".lua": "lua",
        ".pl": "perl",
    }
    return lang_map.get(ext, "text")
