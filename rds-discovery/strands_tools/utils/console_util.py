import io
import os

from rich.console import Console


def create() -> Console:
    """Create rich console instance.

    If STRANDS_TOOL_CONSOLE_MODE environment variable is set to "enabled", output is directed to stdout.

    Returns
        Console instance.
    """
    if os.getenv("STRANDS_TOOL_CONSOLE_MODE") != "enabled":
        return Console(file=io.StringIO())

    return Console()
