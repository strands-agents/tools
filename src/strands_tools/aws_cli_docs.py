"""
AWS CLI Documentation Tools for Command Discovery and Details Retrieval.

This module provides tools to programmatically access AWS CLI documentation,
enabling agents to discover available commands for AWS services and retrieve
detailed documentation for specific commands.

Features:
- Discover all available commands for any AWS service
- Retrieve comprehensive documentation for specific commands
- In-memory caching with configurable TTL to avoid rate limiting
- Support for both standard services and sub-services (e.g., s3api, ec2)

Examples:
    >>> # Get all commands for S3 service
    >>> get_aws_service_commands(service="s3")
    [{"command": "cp", "link": "..."}, {"command": "ls", "link": "..."}, ...]

    >>> # Get detailed documentation for a specific command
    >>> get_aws_command_details(service="s3", command="cp")
    {"command": "cp", "link": "...", "synopsis": "...", "description": "...", ...}
"""

import time
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from strands import tool

# Console for rich output
console = Console()

# Base URL for AWS CLI documentation
AWS_CLI_DOCS_BASE_URL = "https://awscli.amazonaws.com/v2/documentation/api/latest/reference"

# Cache configuration
_cache: Dict[str, Dict[str, Any]] = {}
_cache_ttl_seconds = 3600  # 1 hour default TTL


def _get_cached(key: str) -> Optional[Any]:
    """Get a value from cache if it exists and hasn't expired."""
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["timestamp"] < _cache_ttl_seconds:
            return entry["value"]
        else:
            del _cache[key]
    return None


def _set_cached(key: str, value: Any) -> None:
    """Set a value in cache with current timestamp."""
    _cache[key] = {"value": value, "timestamp": time.time()}


def _fetch_page(url: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch a page from the AWS CLI documentation.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds

    Returns:
        The page HTML content, or None if the request failed
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    except httpx.RequestError:
        return None


def _parse_service_commands(html: str, service: str) -> List[Dict[str, str]]:
    """Parse the service index page to extract available commands.

    Args:
        html: The HTML content of the service index page
        service: The service name for constructing links

    Returns:
        List of dictionaries with command name and documentation link
    """
    soup = BeautifulSoup(html, "html.parser")
    commands = []

    # Find the table of contents or command list
    # AWS CLI docs use different structures, try multiple approaches

    # Approach 1: Look for links in the toctree
    toctree = soup.find("div", class_="toctree-wrapper")
    if toctree:
        for link in toctree.find_all("a", class_="reference internal"):
            command_name = link.get_text(strip=True)
            href = link.get("href", "")
            if command_name and href and not command_name.startswith(service):
                commands.append(
                    {"command": command_name, "link": f"{AWS_CLI_DOCS_BASE_URL}/{service}/{href.rstrip('.html')}.html"}
                )

    # Approach 2: Look for bullet lists with links
    if not commands:
        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                link = li.find("a")
                if link:
                    command_name = link.get_text(strip=True)
                    href = link.get("href", "")
                    if command_name and href and not href.startswith("http"):
                        commands.append(
                            {
                                "command": command_name,
                                "link": f"{AWS_CLI_DOCS_BASE_URL}/{service}/{href.rstrip('.html')}.html",
                            }
                        )

    # Deduplicate and filter
    seen = set()
    unique_commands = []
    for cmd in commands:
        if cmd["command"] not in seen and cmd["command"] != "index":
            seen.add(cmd["command"])
            unique_commands.append(cmd)

    return unique_commands


def _parse_command_details(html: str) -> Dict[str, str]:
    """Parse the command documentation page to extract details.

    Args:
        html: The HTML content of the command documentation page

    Returns:
        Dictionary with command details (synopsis, description, options, examples)
    """
    soup = BeautifulSoup(html, "html.parser")
    details = {}

    # Extract title
    title = soup.find("h1")
    if title:
        details["title"] = title.get_text(strip=True)

    # Extract synopsis
    synopsis_section = soup.find("div", id="synopsis")
    if not synopsis_section:
        synopsis_section = soup.find("section", id="synopsis")
    if synopsis_section:
        pre = synopsis_section.find("pre")
        if pre:
            details["synopsis"] = pre.get_text(strip=True)
        else:
            details["synopsis"] = synopsis_section.get_text(strip=True)

    # Extract description
    desc_section = soup.find("div", id="description")
    if not desc_section:
        desc_section = soup.find("section", id="description")
    if desc_section:
        paragraphs = desc_section.find_all("p")
        details["description"] = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:3])

    # Extract options
    options_section = soup.find("div", id="options")
    if not options_section:
        options_section = soup.find("section", id="options")
    if options_section:
        options = []
        for dt in options_section.find_all("dt"):
            option_name = dt.get_text(strip=True)
            dd = dt.find_next_sibling("dd")
            option_desc = dd.get_text(strip=True)[:200] if dd else ""
            if option_name.startswith("--"):
                options.append(f"{option_name}: {option_desc}")
        details["options"] = options[:15]  # Limit to first 15 options

    # Extract examples
    examples_section = soup.find("div", id="examples")
    if not examples_section:
        examples_section = soup.find("section", id="examples")
    if examples_section:
        examples = []
        for pre in examples_section.find_all("pre"):
            example_text = pre.get_text(strip=True)
            if example_text:
                examples.append(example_text)
        details["examples"] = examples[:5]  # Limit to first 5 examples

    return details


@tool
def get_aws_service_commands(service: str, use_cache: bool = True, cache_ttl: int = 3600) -> str:
    """
    Retrieve a list of available AWS CLI commands for a specific AWS service.

    This tool scrapes the official AWS CLI documentation to discover all available
    commands for the specified service. Results are cached to avoid rate limiting.

    Args:
        service: The AWS service name (e.g., "s3", "ec2", "lambda", "dynamodb").
                 For sub-services, use the full name (e.g., "s3api", "iam").
        use_cache: Whether to use cached results if available. Default: True.
        cache_ttl: Cache time-to-live in seconds. Default: 3600 (1 hour).
                   Only applies when setting new cache entries.

    Returns:
        str: A formatted string containing the list of available commands with
             their documentation links, or an error message if the service
             doesn't exist or cannot be accessed.

    Examples:
        >>> get_aws_service_commands(service="s3")
        "Available commands for 's3' service:
         - cp: https://awscli.amazonaws.com/.../s3/cp.html
         - ls: https://awscli.amazonaws.com/.../s3/ls.html
         ..."

        >>> get_aws_service_commands(service="lambda")
        "Available commands for 'lambda' service:
         - create-function: ...
         - invoke: ...
         ..."

    Raises:
        No exceptions are raised; errors are returned as formatted strings.
    """
    global _cache_ttl_seconds
    _cache_ttl_seconds = cache_ttl

    # Normalize service name
    service = service.lower().strip()
    cache_key = f"service_commands:{service}"

    # Check cache first
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            # Display cached results
            table = Table(
                title=f"🔧 AWS CLI Commands for '{service}' (cached)",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Command", style="green")
            table.add_column("Documentation Link", style="blue")

            for cmd in cached:
                table.add_row(cmd["command"], cmd["link"])

            console.print(table)

            result = f"Available commands for '{service}' service ({len(cached)} commands):\n"
            for cmd in cached:
                result += f"- {cmd['command']}: {cmd['link']}\n"
            return result

    # Fetch the service index page
    url = f"{AWS_CLI_DOCS_BASE_URL}/{service}/index.html"
    html = _fetch_page(url)

    if html is None:
        error_msg = f"Could not find AWS CLI documentation for service '{service}'. "
        error_msg += "Please verify the service name is correct (e.g., 's3', 'ec2', 'lambda')."
        console.print(Panel(error_msg, title="[red]Error", border_style="red"))
        return error_msg

    # Parse the commands
    commands = _parse_service_commands(html, service)

    if not commands:
        error_msg = f"No commands found for service '{service}'. "
        error_msg += "The service may have a different structure or the documentation format changed."
        console.print(Panel(error_msg, title="[yellow]Warning", border_style="yellow"))
        return error_msg

    # Cache the results
    _set_cached(cache_key, commands)

    # Display results
    table = Table(
        title=f"🔧 AWS CLI Commands for '{service}'", box=box.ROUNDED, show_header=True, header_style="bold cyan"
    )
    table.add_column("Command", style="green")
    table.add_column("Documentation Link", style="blue")

    for cmd in commands:
        table.add_row(cmd["command"], cmd["link"])

    console.print(table)

    result = f"Available commands for '{service}' service ({len(commands)} commands):\n"
    for cmd in commands:
        result += f"- {cmd['command']}: {cmd['link']}\n"

    return result


@tool
def get_aws_command_details(service: str, command: str, use_cache: bool = True, cache_ttl: int = 3600) -> str:
    """
    Retrieve detailed documentation for a specific AWS CLI command.

    This tool fetches the full documentation for a specific AWS CLI command,
    including synopsis, description, options, and examples.

    Args:
        service: The AWS service name (e.g., "s3", "ec2", "lambda").
        command: The command name (e.g., "cp", "ls", "describe-instances").
        use_cache: Whether to use cached results if available. Default: True.
        cache_ttl: Cache time-to-live in seconds. Default: 3600 (1 hour).

    Returns:
        str: A formatted string containing the command documentation including
             synopsis, description, options, and examples, or an error message
             if the command doesn't exist.

    Examples:
        >>> get_aws_command_details(service="s3", command="cp")
        "Command: s3 cp
         Synopsis: aws s3 cp <LocalPath> <S3Uri> [--options]
         Description: Copies files between local filesystem and S3...
         Options: --recursive, --exclude, ...
         Examples: aws s3 cp test.txt s3://mybucket/ ..."

        >>> get_aws_command_details(service="ec2", command="describe-instances")
        "Command: ec2 describe-instances
         Synopsis: aws ec2 describe-instances [--instance-ids]...
         ..."

    Raises:
        No exceptions are raised; errors are returned as formatted strings.
    """
    global _cache_ttl_seconds
    _cache_ttl_seconds = cache_ttl

    # Normalize inputs
    service = service.lower().strip()
    command = command.lower().strip()
    cache_key = f"command_details:{service}:{command}"

    # Check cache first
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            # Display cached results
            _display_command_details(service, command, cached, cached.get("link", ""), from_cache=True)
            return _format_command_details(service, command, cached, cached.get("link", ""))

    # Build the documentation URL
    url = f"{AWS_CLI_DOCS_BASE_URL}/{service}/{command}.html"
    html = _fetch_page(url)

    if html is None:
        error_msg = f"Could not find documentation for command '{command}' in service '{service}'. "
        error_msg += "Please verify the command name is correct."
        console.print(Panel(error_msg, title="[red]Error", border_style="red"))
        return error_msg

    # Parse the command details
    details = _parse_command_details(html)
    details["link"] = url

    if not details.get("synopsis") and not details.get("description"):
        error_msg = f"Could not parse documentation for '{service} {command}'. "
        error_msg += "The documentation format may have changed."
        console.print(Panel(error_msg, title="[yellow]Warning", border_style="yellow"))
        return error_msg

    # Cache the results
    _set_cached(cache_key, details)

    # Display and return results
    _display_command_details(service, command, details, url)
    return _format_command_details(service, command, details, url)


def _display_command_details(
    service: str, command: str, details: Dict[str, Any], url: str, from_cache: bool = False
) -> None:
    """Display command details in a rich formatted panel."""
    cache_indicator = " (cached)" if from_cache else ""

    content = []

    if details.get("synopsis"):
        content.append(f"[bold cyan]Synopsis:[/bold cyan]\n{details['synopsis']}")

    if details.get("description"):
        content.append(f"\n[bold cyan]Description:[/bold cyan]\n{details['description']}")

    if details.get("options"):
        options_text = "\n".join(f"  • {opt}" for opt in details["options"][:10])
        content.append(f"\n[bold cyan]Options:[/bold cyan]\n{options_text}")
        if len(details.get("options", [])) > 10:
            content.append(f"  ... and {len(details['options']) - 10} more options")

    if details.get("examples"):
        examples_text = "\n\n".join(details["examples"][:3])
        content.append(f"\n[bold cyan]Examples:[/bold cyan]\n{examples_text}")

    content.append(f"\n[bold cyan]Documentation:[/bold cyan] {url}")

    panel_content = "\n".join(content)
    console.print(
        Panel(
            panel_content,
            title=f"[bold green]📖 aws {service} {command}{cache_indicator}[/bold green]",
            border_style="green",
            expand=False,
        )
    )


def _format_command_details(service: str, command: str, details: Dict[str, Any], url: str) -> str:
    """Format command details as a plain text string."""
    result = f"Command: aws {service} {command}\n"
    result += f"Documentation: {url}\n\n"

    if details.get("synopsis"):
        result += f"Synopsis:\n{details['synopsis']}\n\n"

    if details.get("description"):
        result += f"Description:\n{details['description']}\n\n"

    if details.get("options"):
        result += "Options:\n"
        for opt in details["options"]:
            result += f"  • {opt}\n"
        result += "\n"

    if details.get("examples"):
        result += "Examples:\n"
        for example in details["examples"]:
            result += f"{example}\n\n"

    return result


def clear_cache() -> None:
    """Clear all cached AWS CLI documentation data."""
    global _cache
    _cache = {}
