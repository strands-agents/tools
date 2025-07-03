"""Editor tool designed to do changes iteratively on multiple files.

This module provides a comprehensive file and code editor with rich output formatting,
syntax highlighting, and intelligent text manipulation capabilities. It's designed for
performing iterative changes across multiple files while maintaining a clean interface
and proper error handling.

Key Features:

1. Rich Text Display:
   • Syntax highlighting (Python, JavaScript, Java, HTML, etc.)
   • Line numbering and code formatting
   • Interactive directory trees with icons
   • Beautiful console output with panels and tables

2. File Operations:
   • View: Smart file content display with syntax highlighting
   • Create: New file creation with proper directory handling
   • Replace: Precise string and pattern-based replacement
   • Insert: Smart line finding and content insertion
   • Undo: Automatic backup and restore capability

3. Smart Features:
   • Content History: Caches file contents to reduce reads
   • Pattern Matching: Regex-based replacements
   • Smart Line Finding: Context-aware line location
   • Fuzzy Search: Flexible text matching

4. Safety Features:
   • Automatic backup creation before modifications
   • Content caching for performance
   • Error prevention and validation
   • One-step undo functionality

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import editor

agent = Agent(tools=[editor])

# View a file with syntax highlighting
agent.tool.editor(command="view", path="/path/to/file.py")

# Create a new file
agent.tool.editor(command="create", path="/path/to/file.txt", file_text="Hello World")

# Replace a string in a file
agent.tool.editor(
    command="str_replace",
    path="/path/to/file.py",
    old_str="old text",
    new_str="new text"
)

# Insert text after a line (by number or search text)
agent.tool.editor(
    command="insert",
    path="/path/to/file.py",
    insert_line="def my_function",  # Can be line number or search text
    new_str="    # This is a new comment"
)

# Undo the most recent change
agent.tool.editor(command="undo_edit", path="/path/to/file.py")
```

See the editor function docstring for more details on available commands and parameters.
"""

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from strands import tool

from strands_tools.utils import console_util
from strands_tools.utils.detect_language import detect_language
from strands_tools.utils.user_input import get_user_input

# Global content history cache
CONTENT_HISTORY = {}


def save_content_history(path: str, content: str) -> None:
    """Save file content to history cache."""
    CONTENT_HISTORY[path] = content


def get_last_content(path: str) -> Optional[str]:
    """Get last known content for a file."""
    return CONTENT_HISTORY.get(path)


def find_context_line(content: str, search_text: str, fuzzy: bool = False) -> int:
    """Find line number based on contextual search.

    Args:
        content: File content to search
        search_text: Text to find
        fuzzy: Enable fuzzy matching

    Returns:
        Line number (0-based) or -1 if not found
    """
    lines = content.split("\n")

    if fuzzy:
        # Convert search text to regex pattern
        pattern = ".*".join(map(re.escape, search_text.strip().split()))
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                return i
    else:
        for i, line in enumerate(lines):
            if search_text in line:
                return i

    return -1


def validate_pattern(pattern: str) -> bool:
    """Validate regex pattern."""
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def format_code(code: str, language: str) -> Syntax:
    """Format code using Rich syntax highlighting."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    return syntax


def format_directory_tree(path: str, max_depth: int) -> Tree:
    """Create a Rich tree visualization of directory structure."""
    tree = Tree(f"📁 {os.path.basename(path)}")

    def add_to_tree(current_path: str, tree_node: Tree, depth: int = 0) -> None:
        if depth > max_depth:
            return

        try:
            for item in sorted(os.listdir(current_path)):
                if item.startswith("."):
                    continue

                full_path = os.path.join(current_path, item)
                if os.path.isdir(full_path):
                    branch = tree_node.add(f"📁 {item}")
                    add_to_tree(full_path, branch, depth + 1)
                else:
                    tree_node.add(f"📄 {item}")
        except Exception as e:
            tree_node.add(f"⚠️ Error: {str(e)}")

    add_to_tree(path, tree)
    return tree


def format_output(title: str, content: Any, style: str = "default") -> Panel:
    """Format output with Rich panel."""
    panel = Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED,
        expand=False,
        padding=(1, 2),
    )
    return panel


@tool
def editor(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    insert_line: Optional[Union[str, int]] = None,
    new_str: Optional[str] = None,
    old_str: Optional[str] = None,
    pattern: Optional[str] = None,
    search_text: Optional[str] = None,
    fuzzy: bool = False,
    view_range: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Editor tool designed to do changes iteratively on multiple files.

    This tool provides a comprehensive interface for file operations, including viewing,
    creating, modifying, and searching files with rich output formatting. It features
    syntax highlighting, smart line finding, and automatic backups for safety.

    IMPORTANT ERROR PREVENTION:
    1. Required Parameters:
       • file_text: REQUIRED for 'create' command - content of file to create
       • search_text: REQUIRED for 'find_line' command - text to search
       • insert command: BOTH new_str AND insert_line REQUIRED

    2. Command-Specific Requirements:
       • create: Must provide file_text, file_text is required for create command
       • str_replace: Both old_str and new_str are required for str_replace command
       • pattern_replace: Both pattern and new_str required
       • insert: Both new_str and insert_line required
       • find_line: search_text required

    3. Path Handling:
       • Use absolute paths (e.g., /Users/name/file.txt)
       • Or user-relative paths (~/folder/file.txt)
       • Ensure parent directories exist for create command

    Command Details:
    --------------
    1. view:
       • Displays file content with syntax highlighting
       • Shows directory structure for directory paths
       • Supports viewing specific line ranges with view_range

    2. create:
       • Creates new files with specified content
       • Creates parent directories if they don't exist
       • Caches content for subsequent operations

    3. str_replace:
       • Replaces exact string matches in a file
       • Creates automatic backup before modification
       • Returns details about number of replacements

    4. pattern_replace:
       • Uses regex patterns for advanced text replacement
       • Validates patterns before execution
       • Creates automatic backup before modification

    5. insert:
       • Inserts text after a specified line
       • Supports finding insertion points by line number or search text
       • Shows context around insertion point

    6. find_line:
       • Finds line numbers matching search text
       • Supports fuzzy matching for flexible searches
       • Shows context around found line

    7. undo_edit:
       • Reverts to the most recent backup
       • Removes the backup file after restoration
       • Updates content cache with restored version

    Smart Features:
    ------------
    • Content caching improves performance by reducing file reads
    • Fuzzy search allows finding lines with approximate matches
    • Automatic backups before modifications ensure safety
    • Rich output formatting enhances readability of results

    Args:
        command: The commands to run: `view`, `create`, `str_replace`, `pattern_replace`,
                `insert`, `find_line`, `undo_edit`.
        path: Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
                User paths with tilde (~) are automatically expanded.
        file_text: Required parameter of `create` command, with the content of the file to be created.
        insert_line: Required parameter of `insert` command. The `new_str` will be inserted AFTER
                the line `insert_line` of `path`. Can be a line number or search text.
        new_str: Required parameter containing the new string for `str_replace`,
                `pattern_replace` or `insert` commands.
        old_str: Required parameter of `str_replace` command containing the exact string to replace.
        pattern: Required parameter of `pattern_replace` command containing the regex pattern to match.
        search_text: Text to search for in `find_line` command. Supports fuzzy matching.
        fuzzy: Enable fuzzy matching for `find_line` command.
        view_range: Optional parameter of `view` command. Line range to show [start, end].
                Supports negative indices.

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Response message"}]
        }

        Success case: Returns details about the operation performed
        Error case: Returns information about what went wrong

    Examples:
        1. View a file:
           editor(command="view", path="/path/to/file.py")

        2. Create a new file:
           editor(command="create", path="/path/to/file.txt", file_text="Hello World")

        3. Replace text:
           editor(command="str_replace", path="/path/to/file.py", old_str="old", new_str="new")

        4. Insert after line 10:
           editor(command="insert", path="/path/to/file.py", insert_line=10, new_str="# New line")

        5. Insert after a specific text:
           editor(command="insert", path="/path/to/file.py", insert_line="def main", new_str="    # Comment")

        6. Find a line containing text:
           editor(command="find_line", path="/path/to/file.py", search_text="import os")

        7. Undo recent change:
           editor(command="undo_edit", path="/path/to/file.py")
    """
    console = console_util.create()

    try:
        path = os.path.expanduser(path)

        if not command:
            raise ValueError("Command is required")

        # Validate command
        valid_commands = ["view", "create", "str_replace", "pattern_replace", "insert", "find_line", "undo_edit"]
        if command not in valid_commands:
            raise ValueError(f"Unknown command: {command}. Valid commands: {', '.join(valid_commands)}")

        # Get environment variables at runtime
        editor_dir_tree_max_depth = int(os.getenv("EDITOR_DIR_TREE_MAX_DEPTH", "2"))

        result = ""

        # Check if we're in development mode
        strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

        # For modifying operations, show confirmation dialog unless in BYPASS_TOOL_CONSENT mode
        modifying_commands = {"create", "str_replace", "pattern_replace", "insert"}
        needs_confirmation = command in modifying_commands and not strands_dev

        if needs_confirmation:
            # Show operation preview

            # Preview specific changes for each command
            if command == "create":
                if not file_text:
                    raise ValueError("file_text is required for create command")
                content = file_text
                language = detect_language(path)
                # Use Syntax directly for proper highlighting
                syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                console.print(
                    Panel(
                        syntax,
                        title=f"[bold green]New File Content ({os.path.basename(path)})",
                        border_style="green",
                        box=box.DOUBLE,
                    )
                )
            elif command in {"str_replace", "pattern_replace"}:
                old = old_str if command == "str_replace" else pattern
                new = new_str
                if not old or not new:
                    param_name = "old_str" if command == "str_replace" else "pattern"
                    raise ValueError(f"Both {param_name} and new_str are required for {command} command")
                language = detect_language(path)

                # Create table grid for side-by-side display
                grid = Table.grid(expand=True)
                grid.add_column("Original", justify="left", ratio=1)
                grid.add_column("Arrow", justify="center", width=5)
                grid.add_column("New", justify="left", ratio=1)

                old_panel = Panel(
                    Syntax(
                        str(old),
                        language,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    ),
                    title="[bold red]Original Content",
                    subtitle=f"{len(str(old).splitlines())} lines, {len(str(old))} characters",
                    border_style="red",
                    box=box.ROUNDED,
                )

                new_panel = Panel(
                    Syntax(
                        str(new),
                        language,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    ),
                    title="[bold green]New Content",
                    subtitle=f"{len(str(new).splitlines())} lines, {len(str(new))} characters",
                    border_style="green",
                    box=box.ROUNDED,
                )

                # Add panels with arrow between
                grid.add_row(
                    old_panel,
                    Text("\n\n➔", justify="center", style="bold yellow"),
                    new_panel,
                )

                # Wrap everything in a container panel for consistent look
                preview_panel = Panel(
                    grid,
                    title=f"[bold blue]🔄 Text Replacement Preview ({os.path.basename(path)})",
                    subtitle=f"{os.path.abspath(path)}",
                    border_style="blue",
                    box=box.ROUNDED,
                )

                console.print()
                console.print(preview_panel)
                console.print()
            elif command == "insert":
                if not new_str or insert_line is None:
                    raise ValueError("Both new_str and insert_line are required for insert command")
                language = detect_language(path)
                # Create table with syntax highlighting
                table = Table(title="Insertion Preview", show_header=True)
                table.add_column("Target Line", style="yellow")
                table.add_column("Content to Insert", style="green")
                table.add_row(
                    str(insert_line),
                    Syntax(new_str, language, theme="monokai", line_numbers=True),
                )
                console.print(table)

            # Get user confirmation
            user_input = get_user_input(
                f"<yellow><bold>Do you want to proceed with the {command} operation?</bold> [y/*]</yellow>"
            )
            if user_input.lower().strip() != "y":
                cancellation_reason = (
                    user_input
                    if user_input.strip() != "n"
                    else get_user_input("Please provide a reason for cancellation:")
                )
                error_message = f"Operation cancelled by the user. Reason: {cancellation_reason}"
                error_panel = Panel(
                    Text(error_message, style="bold blue"),
                    title="[bold blue]Operation Cancelled",
                    border_style="blue",
                    box=box.HEAVY,
                    expand=False,
                )
                console.print(error_panel)
                return {
                    "status": "error",
                    "content": [{"text": error_message}],
                }

        if command == "view":
            if os.path.isfile(path):
                # Check content history first
                content = get_last_content(path)
                if content is None:
                    with open(path, "r") as f:
                        content = f.read()
                    save_content_history(path, content)

                if view_range:
                    lines = content.split("\n")
                    start = max(0, view_range[0] - 1)
                    end = min(len(lines), view_range[1])
                    content = "\n".join(lines[start:end])

                # Determine file type for syntax highlighting
                file_ext = os.path.splitext(path)[1].lower()
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
                }
                language = lang_map.get(file_ext, "text")

                # Format and print the content
                formatted = format_code(content, language)
                formatted_output = format_output(f"📄 File: {os.path.basename(path)}", formatted, "green")
                console.print(formatted_output)
                result = f"File content displayed in console.\nContent: {content}"

            elif os.path.isdir(path):
                # Directory visualization
                tree = format_directory_tree(path, editor_dir_tree_max_depth)
                formatted_output = format_output(f"📁 Directory: {path}", tree, "blue")
                console.print(formatted_output)
                result = f"Directory structure displayed in console.\nDirectory tree: {path}"
            else:
                raise ValueError(f"Path {path} does not exist")

        elif command == "create":
            if not file_text:
                raise ValueError("file_text is required for create command")

            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Write the file and cache content
            with open(path, "w") as f:
                f.write(file_text)
            save_content_history(path, file_text)

            # Just return success message
            result = f"File {path} created successfully"

        elif command == "str_replace":
            if not old_str or not new_str:
                raise ValueError("Both old_str and new_str are required for str_replace command")

            # Check content history first
            content = get_last_content(path)
            if content is None:
                with open(path, "r") as f:
                    content = f.read()
                save_content_history(path, content)

            # Count occurrences
            count = content.count(old_str)
            if count == 0:
                # Return existing content if no matches
                return {
                    "status": "error",
                    "content": [{"text": f"Note: old_str not found in {path}. Current content:\n{content}"}],
                }

            # Make replacements and backup
            new_content = content.replace(old_str, new_str)
            backup_path = f"{path}.bak"
            shutil.copy2(path, backup_path)

            # Write new content and update cache
            with open(path, "w") as f:
                f.write(new_content)
            save_content_history(path, new_content)

            result = (
                f"Text replacement complete and details displayed in console.\nFile: {path}\n"
                f"Replaced {count} occurrence{'s' if count > 1 else ''}\n"
                f"Old string: {old_str}\nNew string: {new_str}\n"
            )

        elif command == "pattern_replace":
            if not pattern or not new_str:
                raise ValueError("Both pattern and new_str are required for pattern_replace command")

            # Validate pattern
            if not validate_pattern(pattern):
                raise ValueError(f"Invalid regex pattern: {pattern}")

            # Check content history
            content = get_last_content(path)
            if content is None:
                with open(path, "r") as f:
                    content = f.read()
                save_content_history(path, content)

            # Compile pattern and find matches
            regex = re.compile(pattern)
            matches = list(regex.finditer(content))
            if not matches:
                return {
                    "status": "success",
                    "content": [{"text": f"Note: pattern '{pattern}' not found in {path}. Current content:{content}"}],
                }

            # Create preview table with match context
            preview_table = Table(
                title="📝 Match Preview",
                show_header=True,
                header_style="bold magenta",
                border_style="blue",
            )
            preview_table.add_column("Context", style="dim")
            preview_table.add_column("Match", style="bold yellow")
            preview_table.add_column("→", style="green")
            preview_table.add_column("Replacement", style="bold green")

            # Add match previews with context
            for match in matches[:5]:  # Show first 5 matches
                start, end = match.span()
                context_start = max(0, start - 20)
                context_end = min(len(content), end + 20)

                before = content[context_start:start]
                matched = content[start:end]

                # Highlight the replacement
                preview = regex.sub(new_str, matched)

                preview_table.add_row(f"...{before}", matched, "→", f"{preview}...")

            # Show more indicator if needed
            if len(matches) > 5:
                preview_table.add_row("...", f"({len(matches) - 5} more matches)", "→", "...")

            # Make replacements and backup
            new_content = regex.sub(new_str, content)
            backup_path = f"{path}.bak"
            shutil.copy2(path, backup_path)

            # Write new content and update cache
            with open(path, "w") as f:
                f.write(new_content)
            save_content_history(path, new_content)

            # Show summary info
            info_table = Table(show_header=False, border_style="blue")
            info_table.add_column("", style="cyan")
            info_table.add_column("", style="white")

            info_table.add_row("Pattern:", pattern)
            info_table.add_row("Replacement:", new_str)
            info_table.add_row("Total Matches:", str(len(matches)))
            info_table.add_row("File:", path)
            info_table.add_row("Backup:", backup_path)

            # Render the UI
            console.print("")
            console.print(Panel(info_table, title="ℹ️ Pattern Replace Summary", border_style="blue"))
            console.print("")
            console.print(preview_table)
            console.print("")
            console.print(
                Panel(
                    "✅ Changes applied successfully! Use 'undo_edit' to revert if needed.",
                    border_style="green",
                )
            )

            result = (
                f"Pattern replacement complete and details displayed in console.\nFile: {path}\n"
                f"Pattern: {pattern}\nNew string: {new_str}\nMatches: {len(matches)}"
            )

        elif command == "insert":
            if not new_str or insert_line is None:
                raise ValueError("Both new_str and insert_line are required for insert command")

            # Get content
            content = get_last_content(path)
            if content is None:
                with open(path, "r") as f:
                    content = f.read()
                save_content_history(path, content)

            lines = content.split("\n")

            # Handle string-based line finding
            if isinstance(insert_line, str):
                line_num = find_context_line(content, insert_line, fuzzy)
                if line_num == -1:
                    return {
                        "status": "success",
                        "content": [
                            {
                                "text": (
                                    f"Note: Could not find insertion point '{insert_line}' in {path}. "
                                    f"Current content:\n{content}"
                                )
                            }
                        ],
                    }
                insert_line = line_num

            # Validate line number
            if insert_line < 0 or insert_line > len(lines):
                raise ValueError(f"insert_line {insert_line} is out of range")

            # Make backup
            backup_path = f"{path}.bak"
            shutil.copy2(path, backup_path)

            # Insert and write
            lines.insert(insert_line, new_str)
            new_content = "\n".join(lines)
            with open(path, "w") as f:
                f.write(new_content)
            save_content_history(path, new_content)

            # Show context
            context_start = max(0, insert_line - 2)
            context_end = min(len(lines), insert_line + 3)
            context_lines = lines[context_start:context_end]

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Line", style="cyan", justify="right")
            table.add_column("Content", style="white")

            for i, line in enumerate(context_lines, start=context_start + 1):
                style = "green" if i == insert_line + 1 else "white"
                table.add_row(str(i), line.rstrip(), style=style)

            formatted_output = format_output(
                "➕ Text Insertion Complete",
                f"File: {path}\nInserted at line {insert_line}\n",
                "green",
            )
            console.print(formatted_output)
            console.print(table)
            result = (
                f"Text insertion complete and details displayed in console.\nFile: {path}\n"
                f"Inserted at line {insert_line}"
            )

        elif command == "find_line":
            if not search_text:
                raise ValueError("search_text is required for find_line command")

            # Get content
            content = get_last_content(path)
            if content is None:
                with open(path, "r") as f:
                    content = f.read()
                save_content_history(path, content)

            # Find line
            line_num = find_context_line(content, search_text, fuzzy)

            if line_num == -1:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Note: Could not find '{search_text}' in {path} while using editor tool, "
                                f"to correct next step, here's the current content of file:\n{content}\n"
                            )
                        }
                    ],
                }

            # Show context
            lines = content.split("\n")
            context_start = max(0, line_num - 2)
            context_end = min(len(lines), line_num + 3)
            context_lines = lines[context_start:context_end]

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Line", style="cyan", justify="right")
            table.add_column("Content", style="white")

            for i, line in enumerate(context_lines, start=context_start + 1):
                style = "green" if i == line_num + 1 else "white"
                table.add_row(str(i), line.rstrip(), style=style)

            formatted_output = format_output(
                "🔍 Line Found",
                f"File: {path}\nFound at line {line_num + 1}\n",
                "green",
            )
            console.print(formatted_output)
            console.print(table)
            result = f"Line found in file.\nFile: {path}\nLine number: {line_num + 1}"

        elif command == "undo_edit":
            backup_path = f"{path}.bak"

            if not os.path.exists(backup_path):
                raise ValueError(f"No backup file found for {path}")

            # Restore from backup
            shutil.copy2(backup_path, path)
            os.remove(backup_path)

            # Update cache from backup
            with open(path, "r") as f:
                content = f.read()
            save_content_history(path, content)

            formatted_output = format_output("↩️ Undo Complete", f"Successfully reverted changes to {path}", "yellow")
            console.print(formatted_output)
            result = f"Successfully reverted changes to {path}"

        else:
            raise ValueError(f"Unknown command: {command}")

        return {
            "status": "success",
            "content": [{"text": result}],
        }

    except Exception as e:
        error_msg = format_output("❌ Error", str(e), "red")
        console.print(error_msg)
        return {
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
