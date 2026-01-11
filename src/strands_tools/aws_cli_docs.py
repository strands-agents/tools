"""
AWS CLI Documentation Tools for command discovery and details retrieval.

This module provides programmatic access to official AWS CLI documentation,
enabling agents to discover available commands and retrieve detailed documentation.

Key Features:
- Discover all available AWS CLI commands for any service
- Retrieve comprehensive documentation including synopsis, options, and examples
- In-memory caching with configurable TTL to avoid rate limiting
- Rich console output for better readability

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import aws_cli_docs

agent = Agent(tools=[aws_cli_docs.get_aws_service_commands, aws_cli_docs.get_aws_command_details])

# Discover S3 commands
result = agent.tool.get_aws_service_commands(service="s3")

# Get details for a specific command
result = agent.tool.get_aws_command_details(service="s3", command="cp")
```

References:
- AWS CLI Documentation: https://awscli.amazonaws.com/v2/documentation/api/latest/reference/
- Feature Request: https://github.com/strands-agents/tools/issues/352
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from markdownify import markdownify as md
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from strands import tool

logger = logging.getLogger(__name__)

# AWS CLI Documentation base URL
AWS_CLI_DOCS_BASE_URL = "https://awscli.amazonaws.com/v2/documentation/api/latest/reference"

# Initialize Rich console
console = Console()

# Cache configuration
DEFAULT_CACHE_TTL = int(os.getenv("AWS_CLI_DOCS_CACHE_TTL", "3600"))  # 1 hour default


@dataclass
class CacheEntry:
    """Cache entry with timestamp for TTL management."""

    data: Any
    timestamp: float = field(default_factory=time.time)

    def is_expired(self, ttl: int) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > ttl


class AwsCliDocsCache:
    """Simple in-memory cache for AWS CLI documentation."""

    def __init__(self, ttl: int = DEFAULT_CACHE_TTL):
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        entry = self._cache.get(key)
        if entry and not entry.is_expired(self._ttl):
            logger.debug(f"Cache hit for: {key}")
            return entry.data
        if entry:
            # Clean up expired entry
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cache value."""
        self._cache[key] = CacheEntry(data=value)
        logger.debug(f"Cached: {key}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


# Global cache instance
_cache = AwsCliDocsCache()


def _fetch_page(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch a page from the AWS CLI documentation.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds

    Returns:
        The page HTML content or None if request failed
    """
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching {url}: {e}")
        return None


def _extract_text_from_html(html: str, selector_pattern: str = None) -> str:
    """Extract clean text from HTML using regex (no BeautifulSoup dependency).

    Args:
        html: HTML content
        selector_pattern: Optional regex pattern to match specific sections

    Returns:
        Cleaned text content
    """
    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert to markdown for better readability
    try:
        text = md(html, strip=["script", "style", "nav", "header", "footer"])
    except Exception:
        # Fallback: strip all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_service_commands(html: str, service: str) -> List[Dict[str, str]]:
    """Parse the service index page to extract available commands.

    Args:
        html: The HTML content of the service index page
        service: The service name for building full links

    Returns:
        List of command dictionaries with 'command' and 'link' keys
    """
    commands = []
    seen_commands = set()

    # Pattern 1: Look for toctree links (most common pattern in AWS CLI docs)
    # Match href="command.html" or href="command/index.html"
    toctree_pattern = r'class="toctree-l1"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    for match in re.finditer(toctree_pattern, html, re.IGNORECASE):
        href, text = match.groups()
        command_name = href.replace(".html", "").replace("/index", "").strip("/")
        if command_name and command_name not in ("index", "") and command_name not in seen_commands:
            full_link = f"{AWS_CLI_DOCS_BASE_URL}/{service}/{href}"
            commands.append({"command": command_name, "link": full_link})
            seen_commands.add(command_name)

    # Pattern 2: Alternative - look for any links to .html files in lists
    if not commands:
        link_pattern = r'<li[^>]*>\s*<a[^>]*href="([^"]+\.html)"[^>]*>([^<]+)</a>'
        for match in re.finditer(link_pattern, html, re.IGNORECASE):
            href, text = match.groups()
            # Skip if it's a parent directory reference or index
            if "../" in href or href == "index.html":
                continue
            command_name = href.replace(".html", "")
            if command_name and command_name not in seen_commands:
                full_link = f"{AWS_CLI_DOCS_BASE_URL}/{service}/{href}"
                commands.append({"command": command_name, "link": full_link})
                seen_commands.add(command_name)

    return commands


def _parse_command_details(html: str) -> Dict[str, str]:
    """Parse a command documentation page to extract details.

    Args:
        html: The HTML content of the command documentation page

    Returns:
        Dictionary with 'synopsis', 'description', 'options', 'examples' keys
    """
    details = {}

    # Extract Description section
    desc_match = re.search(
        r'<div[^>]*id="description"[^>]*>(.*?)</div>\s*(?:<div|$)', html, re.DOTALL | re.IGNORECASE
    )
    if desc_match:
        desc_html = desc_match.group(1)
        # Get first few paragraphs
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", desc_html, re.DOTALL | re.IGNORECASE)
        if paragraphs:
            desc_text = " ".join(re.sub(r"<[^>]+>", "", p).strip() for p in paragraphs[:3])
            details["description"] = desc_text

    # Extract Synopsis section
    synopsis_match = re.search(
        r'<div[^>]*id="synopsis"[^>]*>(.*?)</div>\s*(?:<div|$)', html, re.DOTALL | re.IGNORECASE
    )
    if synopsis_match:
        synopsis_html = synopsis_match.group(1)
        # Look for pre or code blocks
        pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", synopsis_html, re.DOTALL | re.IGNORECASE)
        if pre_match:
            details["synopsis"] = re.sub(r"<[^>]+>", "", pre_match.group(1)).strip()
        else:
            code_match = re.search(r"<code[^>]*>(.*?)</code>", synopsis_html, re.DOTALL | re.IGNORECASE)
            if code_match:
                details["synopsis"] = re.sub(r"<[^>]+>", "", code_match.group(1)).strip()

    # Extract Options section
    options_match = re.search(
        r'<div[^>]*id="options"[^>]*>(.*?)</div>\s*(?:<div[^>]*id=|$)', html, re.DOTALL | re.IGNORECASE
    )
    if options_match:
        options_html = options_match.group(1)
        # Find dt/dd pairs (definition list format)
        options_text = []
        dt_dd_pattern = r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>"
        for match in re.finditer(dt_dd_pattern, options_html, re.DOTALL | re.IGNORECASE):
            opt_name = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            opt_desc = re.sub(r"<[^>]+>", "", match.group(2)).strip()[:200]
            if opt_name:
                options_text.append(f"{opt_name}: {opt_desc}...")
        if options_text:
            details["options"] = "\n".join(options_text[:15])  # Limit to first 15 options

    # Extract Examples section
    examples_match = re.search(
        r'<div[^>]*id="examples"[^>]*>(.*?)</div>\s*(?:<div[^>]*id=|$)', html, re.DOTALL | re.IGNORECASE
    )
    if examples_match:
        examples_html = examples_match.group(1)
        examples = []
        for pre_match in re.finditer(r"<pre[^>]*>(.*?)</pre>", examples_html, re.DOTALL | re.IGNORECASE):
            example_text = re.sub(r"<[^>]+>", "", pre_match.group(1)).strip()
            if example_text:
                examples.append(example_text)
        if examples:
            details["examples"] = "\n\n---\n\n".join(examples[:3])  # Limit to first 3 examples

    # If we couldn't parse sections, try to get full text
    if not details:
        # Convert HTML to markdown for better readability
        try:
            full_text = md(html, strip=["script", "style", "nav", "header", "footer"])
            # Truncate if too long
            if len(full_text) > 5000:
                full_text = full_text[:5000] + "..."
            details["full_text"] = full_text
        except Exception:
            # Fallback: strip all HTML tags
            full_text = re.sub(r"<[^>]+>", " ", html)
            full_text = re.sub(r"\s+", " ", full_text).strip()
            if len(full_text) > 5000:
                full_text = full_text[:5000] + "..."
            details["full_text"] = full_text

    return details


def _format_commands_table(service: str, commands: List[Dict[str, str]]) -> Panel:
    """Format commands list as a rich table for console output."""
    table = Table(title=f"AWS CLI Commands for '{service}'", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Documentation Link", style="blue")

    for cmd in commands[:30]:  # Limit display to first 30
        table.add_row(cmd["command"], cmd["link"])

    if len(commands) > 30:
        table.add_row("...", f"({len(commands) - 30} more commands)")

    return Panel(table, title=f"[bold green]Found {len(commands)} commands", border_style="green")


def _format_command_details(service: str, command: str, link: str, details: Dict[str, str]) -> Panel:
    """Format command details as a rich panel for console output."""
    content_parts = [
        f"[bold]Service:[/bold] {service}",
        f"[bold]Command:[/bold] {command}",
        f"[bold]Link:[/bold] {link}\n",
    ]

    if "synopsis" in details:
        content_parts.append(f"[bold cyan]Synopsis:[/bold cyan]\n{details['synopsis']}\n")

    if "description" in details:
        content_parts.append(f"[bold cyan]Description:[/bold cyan]\n{details['description']}\n")

    if "options" in details:
        content_parts.append(f"[bold cyan]Options (partial):[/bold cyan]\n{details['options']}\n")

    if "examples" in details:
        content_parts.append(f"[bold cyan]Examples:[/bold cyan]\n{details['examples']}")

    if "full_text" in details:
        content_parts.append(f"[bold cyan]Documentation:[/bold cyan]\n{details['full_text']}")

    return Panel("\n".join(content_parts), title=f"[bold blue]aws {service} {command}", border_style="blue")


@tool
def get_aws_service_commands(service: str, use_cache: bool = True) -> str:
    """
    Retrieve a list of available AWS CLI commands for a specified AWS service.

    This tool fetches the official AWS CLI documentation for the given service
    and returns a structured list of all available commands with their documentation links.

    Args:
        service: The AWS service name (e.g., "s3", "ec2", "lambda", "dynamodb", "rds").
                 Use lowercase names as they appear in AWS CLI (e.g., "s3" not "S3").
        use_cache: Whether to use cached results if available. Defaults to True.
                   Set to False to force fresh fetch from AWS documentation.

    Returns:
        A string containing a JSON-formatted list of commands, where each command has:
        - command: The command name (e.g., "cp", "ls", "create-bucket")
        - link: Full URL to the command's documentation page

        Returns an error message if the service is not found or request fails.

    Examples:
        >>> get_aws_service_commands(service="s3")
        [{"command": "cp", "link": "https://..."}, {"command": "ls", "link": "https://..."}, ...]

        >>> get_aws_service_commands(service="lambda")
        [{"command": "invoke", "link": "https://..."}, {"command": "create-function", "link": "https://..."}, ...]

    Use Cases:
        - Discovering what operations are available for a specific AWS service
        - Exploring AWS CLI capabilities before suggesting commands to users
        - Building command suggestions based on available options
    """
    # Normalize service name
    service = service.lower().strip()

    # Check cache first
    cache_key = f"service_commands:{service}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached:
            console.print(f"[dim]Using cached results for '{service}'[/dim]")
            console.print(_format_commands_table(service, cached))
            return json.dumps(cached, indent=2)

    # Build URL for service index
    url = f"{AWS_CLI_DOCS_BASE_URL}/{service}/index.html"
    logger.info(f"Fetching AWS CLI docs for service '{service}' from {url}")

    # Fetch the page
    html = _fetch_page(url)
    if not html:
        error_msg = (
            f"Could not fetch AWS CLI documentation for service '{service}'. "
            "The service may not exist or the documentation server is unavailable."
        )
        console.print(f"[red]{error_msg}[/red]")
        return json.dumps({"error": error_msg})

    # Parse commands
    commands = _parse_service_commands(html, service)

    if not commands:
        # Try alternative URL format (some services use different structure)
        alt_url = f"{AWS_CLI_DOCS_BASE_URL}/{service}.html"
        html = _fetch_page(alt_url)
        if html:
            commands = _parse_service_commands(html, service)

    if not commands:
        error_msg = (
            f"No commands found for service '{service}'. "
            "Please verify the service name is correct (use lowercase, e.g., 's3', 'ec2', 'lambda')."
        )
        console.print(f"[yellow]{error_msg}[/yellow]")
        return json.dumps({"error": error_msg})

    # Cache the results
    _cache.set(cache_key, commands)

    # Display formatted output
    console.print(_format_commands_table(service, commands))

    return json.dumps(commands, indent=2)


@tool
def get_aws_command_details(service: str, command: str, use_cache: bool = True) -> str:
    """
    Retrieve comprehensive documentation for a specific AWS CLI command.

    This tool fetches detailed documentation for the specified command including
    synopsis, description, available options, and usage examples.

    Args:
        service: The AWS service name (e.g., "s3", "ec2", "lambda").
                 Use lowercase names as they appear in AWS CLI.
        command: The specific command name (e.g., "cp", "describe-instances", "invoke").
                 Use the exact command name as listed in AWS CLI documentation.
        use_cache: Whether to use cached results if available. Defaults to True.

    Returns:
        A string containing a JSON object with:
        - service: The AWS service name
        - command: The command name
        - link: Full URL to the documentation page
        - details: An object containing:
            - synopsis: Command syntax and usage pattern
            - description: What the command does
            - options: Available parameters and their descriptions
            - examples: Usage examples from official documentation

        Returns an error message if the command is not found.

    Examples:
        >>> get_aws_command_details(service="s3", command="cp")
        {
            "service": "s3",
            "command": "cp",
            "link": "https://...",
            "details": {
                "synopsis": "aws s3 cp <LocalPath> <S3Uri>...",
                "description": "Copies a file...",
                "options": "--recursive: ...",
                "examples": "aws s3 cp test.txt s3://mybucket/..."
            }
        }

    Use Cases:
        - Getting exact syntax for an AWS CLI command
        - Understanding available options and parameters
        - Finding usage examples to construct proper commands
        - Troubleshooting command errors by checking correct parameters
    """
    # Normalize inputs
    service = service.lower().strip()
    command = command.lower().strip()

    # Check cache
    cache_key = f"command_details:{service}:{command}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached:
            console.print(f"[dim]Using cached results for '{service} {command}'[/dim]")
            console.print(_format_command_details(service, command, cached["link"], cached["details"]))
            return json.dumps(cached, indent=2)

    # Build URL for command documentation
    url = f"{AWS_CLI_DOCS_BASE_URL}/{service}/{command}.html"
    logger.info(f"Fetching AWS CLI docs for '{service} {command}' from {url}")

    # Fetch the page
    html = _fetch_page(url)
    if not html:
        # Try with index.html for subcommand groups
        url = f"{AWS_CLI_DOCS_BASE_URL}/{service}/{command}/index.html"
        html = _fetch_page(url)

    if not html:
        error_msg = (
            f"Could not fetch documentation for 'aws {service} {command}'. "
            f"The command may not exist. Use get_aws_service_commands(service='{service}') to see available commands."
        )
        console.print(f"[red]{error_msg}[/red]")
        return json.dumps({"error": error_msg})

    # Parse the details
    details = _parse_command_details(html)

    if not details:
        error_msg = f"Could not parse documentation for 'aws {service} {command}'. The page structure may have changed."
        console.print(f"[yellow]{error_msg}[/yellow]")
        return json.dumps({"error": error_msg})

    # Build result
    result = {"service": service, "command": command, "link": url, "details": details}

    # Cache the result
    _cache.set(cache_key, result)

    # Display formatted output
    console.print(_format_command_details(service, command, url, details))

    return json.dumps(result, indent=2)


# Export functions for module-level access
__all__ = ["get_aws_service_commands", "get_aws_command_details"]
