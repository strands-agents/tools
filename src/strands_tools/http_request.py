"""
Make HTTP requests with comprehensive authentication, session management, and metrics.
Supports all major authentication types and enterprise patterns.

Environment Variable Support:
1. Authentication tokens:
   - Uses auth_env_var parameter to read tokens from environment (e.g., GITHUB_TOKEN, GITLAB_TOKEN)
   - Example: http_request(method="GET", url="...", auth_type="token", auth_env_var="GITHUB_TOKEN")
   - Supported variables: GITHUB_TOKEN, GITLAB_TOKEN, SLACK_BOT_TOKEN, AWS_ACCESS_KEY_ID, etc.
2. AWS credentials:
   - Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION automatically
   - Example: http_request(method="GET", url="...", auth_type="aws_sig_v4", aws_auth={"service": "s3"})
Use the environment tool (agent.tool.environment) to view available environment variables:
- List all: environment(action="list")
- Get specific: environment(action="get", name="GITHUB_TOKEN")
- Set new: environment(action="set", name="CUSTOM_TOKEN", value="your-token")
"""

import base64
import collections
import datetime
import http.cookiejar
import json
import os
import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import markdownify
import requests
from aws_requests_auth.aws_auth import AWSRequestsAuth
from requests.adapters import HTTPAdapter
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from strands.types.tools import (
    ToolResult,
    ToolUse,
)
from urllib3 import Retry

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

TOOL_SPEC = {
    "name": "http_request",
    "description": (
        "Make HTTP requests to any API with comprehensive authentication including Bearer tokens, Basic auth, "
        "JWT, AWS SigV4, Digest auth, and enterprise authentication patterns. Automatically reads tokens from "
        "environment variables (GITHUB_TOKEN, GITLAB_TOKEN, AWS credentials, etc.) when auth_env_var is specified. "
        "Use environment(action='list') to view available variables. Includes session management, metrics, "
        "streaming support, cookie handling, redirect control, proxy support, and optional HTML to markdown conversion."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, etc.)",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to",
                },
                "auth_type": {
                    "type": "string",
                    "enum": [
                        "Bearer",
                        "token",
                        "basic",
                        "digest",
                        "jwt",
                        "aws_sig_v4",
                        "kerberos",
                        "custom",
                        "api_key",
                    ],
                    "description": "Authentication type to use",
                },
                "auth_token": {
                    "type": "string",
                    "description": "Authentication token (if not provided, will check environment variables)",
                },
                "auth_env_var": {
                    "type": "string",
                    "description": "Name of environment variable containing the auth token",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers as key-value pairs",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST, PUT, etc.)",
                },
                "verify_ssl": {
                    "type": "boolean",
                    "description": "Whether to verify SSL certificates",
                },
                "cookie": {
                    "type": "string",
                    "description": "Path to cookie file to use for the request",
                },
                "cookie_jar": {
                    "type": "string",
                    "description": "Path to cookie jar file to save cookies to",
                },
                "session_config": {
                    "type": "object",
                    "description": "Session configuration (cookies, keep-alive, etc)",
                    "properties": {
                        "keep_alive": {"type": "boolean"},
                        "max_retries": {"type": "integer"},
                        "pool_size": {"type": "integer"},
                        "cookie_persistence": {"type": "boolean"},
                    },
                },
                "metrics": {
                    "type": "boolean",
                    "description": "Whether to collect request metrics",
                },
                "streaming": {
                    "type": "boolean",
                    "description": "Enable streaming response handling",
                },
                "allow_redirects": {
                    "type": "boolean",
                    "description": "Whether to follow redirects (default: True)",
                },
                "max_redirects": {
                    "type": "integer",
                    "description": "Maximum number of redirects to follow (default: 30)",
                },
                "convert_to_markdown": {
                    "type": "boolean",
                    "description": "Convert HTML responses to markdown format (default: False).",
                },
                "aws_auth": {
                    "type": "object",
                    "description": "AWS auth configuration for SigV4",
                    "properties": {
                        "service": {"type": "string"},
                        "region": {"type": "string"},
                        "access_key": {"type": "string"},
                        "secret_key": {"type": "string"},
                        "session_token": {"type": "string"},
                        "refresh_credentials": {"type": "boolean"},
                    },
                },
                "basic_auth": {
                    "type": "object",
                    "description": "Basic auth credentials",
                    "properties": {
                        "username": {"type": "string"},
                        "password": {"type": "string"},
                    },
                    "required": ["username", "password"],
                },
                "digest_auth": {
                    "type": "object",
                    "description": "Digest auth credentials",
                    "properties": {
                        "username": {"type": "string"},
                        "password": {"type": "string"},
                        "realm": {"type": "string"},
                    },
                },
                "jwt_config": {
                    "type": "object",
                    "description": "JWT configuration",
                    "properties": {
                        "secret": {"type": "string"},
                        "algorithm": {"type": "string"},
                        "expiry": {"type": "integer"},
                    },
                },
                "proxies": {
                    "type": "object",
                    "description": "Dictionary mapping protocol or protocol and hostname to the URL of the proxy.",
                    "properties": {
                        "http": {"type": "string"},
                        "https": {"type": "string"},
                        "ftp": {"type": "string"},
                    },
                },
            },
            "required": ["method", "url"],
        }
    },
}

# Session cache keyed by domain
SESSION_CACHE = {}

# Metrics storage
REQUEST_METRICS = collections.defaultdict(list)


def extract_content_from_html(html: str) -> str:
    """Convert HTML content to Markdown format.

    Args:
        html: Raw HTML content to process

    Returns:
        Markdown version of the content, or original HTML if conversion fails
    """
    try:
        content = markdownify.markdownify(
            html,
            heading_style=markdownify.ATX,
        )
        return content
    except Exception:
        # If conversion fails, return original HTML
        return html


def create_session(config: Dict[str, Any]) -> requests.Session:
    """Create and configure a requests Session object."""
    session = requests.Session()

    if config.get("keep_alive", True):
        adapter = HTTPAdapter(
            pool_connections=config.get("pool_size", 10),
            pool_maxsize=config.get("pool_size", 10),
            max_retries=Retry(
                total=config.get("max_retries", 3),
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

    if not config.get("cookie_persistence", True):
        session.cookies.clear()

    return session


def get_cached_session(url: str, config: Dict[str, Any]) -> requests.Session:
    """Get or create a cached session for the domain."""
    domain = urlparse(url).netloc
    if domain not in SESSION_CACHE:
        SESSION_CACHE[domain] = create_session(config)
    return SESSION_CACHE[domain]


def process_metrics(start_time: float, response: requests.Response) -> Dict[str, Any]:
    """Process and store request metrics."""
    end_time = time.time()
    metrics = {
        "duration": round(end_time - start_time, 3),
        "status_code": response.status_code,
        "bytes_sent": (len(response.request.body) if response.request and response.request.body is not None else 0),
        "bytes_received": len(response.content),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    REQUEST_METRICS[urlparse(response.url).netloc].append(metrics)
    return metrics


def handle_basic_auth(username: str, password: str) -> Dict[str, str]:
    """Process Basic authentication."""
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}


def handle_digest_auth(config: Dict[str, Any], method: str, url: str) -> requests.auth.HTTPDigestAuth:
    """Set up Digest authentication."""
    return requests.auth.HTTPDigestAuth(config["username"], config["password"])


def get_aws_credentials() -> tuple:
    """Get AWS credentials from boto3 with proper credential chain."""
    import boto3

    # Create a boto3 session to ensure we're using the same credential chain
    session = boto3.Session()
    credentials = session.get_credentials()

    if not credentials:
        raise ValueError("No AWS credentials found in the credential chain")

    frozen = credentials.get_frozen_credentials()
    return frozen, session.region_name


def handle_aws_sigv4(config: Dict[str, Any], url: str) -> AWSRequestsAuth:
    """
    Configure AWS SigV4 authentication using boto3's credential chain.
    """
    try:
        # Get credentials using boto3's credential chain
        credentials, default_region = get_aws_credentials()

        # Get service from config (required)
        service = config["service"]

        # Get region from config or use default
        region = config.get("region") or default_region

        if not region:
            raise ValueError("AWS region not found in config or environment")

        parsed = urlparse(url)
        auth = AWSRequestsAuth(
            aws_access_key=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_host=parsed.netloc,
            aws_region=region,
            aws_service=service,
            aws_token=credentials.token,  # Add session token directly
        )

        return auth

    except Exception as e:
        raise ValueError(f"AWS authentication error: {str(e)}") from e


def handle_jwt(config: Dict[str, Any]) -> Dict[str, str]:
    """Process JWT authentication."""
    try:
        import jwt  # Imported here to avoid global dependency
    except ImportError:
        raise ImportError(
            "ImportError: PyJWT package is required for JWT authentication. Install with: pip install PyJWT"
        ) from None

    # Create expiration time using datetime module properly
    expiry_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=config["expiry"])
    token = jwt.encode(
        {"exp": expiry_time},
        config["secret"],
        algorithm=config["algorithm"],
    )

    # Convert token to string based on type
    token_str = token.decode("utf-8") if hasattr(token, "decode") else str(token)

    return {"Authorization": f"Bearer {token_str}"}


def format_json_response(content: str) -> Union[str, Syntax]:
    """Format JSON response with syntax highlighting if valid JSON."""
    try:
        parsed = json.loads(content)
        formatted = json.dumps(parsed, indent=2)
        return Syntax(formatted, "json", theme="monokai", line_numbers=False)
    except BaseException:
        return content


def format_headers_table(headers: Dict) -> Table:
    """Format headers as a rich table."""
    table = Table(title="Response Headers", show_header=True, box=box.ROUNDED)
    table.add_column("Header", style="cyan")
    table.add_column("Value", style="green")

    for key, value in headers.items():
        # Truncate very long header values
        if isinstance(value, str) and len(value) > 100:
            value = f"{value[:100]}..."
        table.add_row(key, str(value))

    return table


def process_auth_headers(headers: Dict[str, Any], tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process authentication headers based on input parameters.

    Supports multiple authentication methods:
    1. Environment variables: Uses auth_env_var to read tokens
    2. Direct token: Uses auth_token parameter

    Special handling for different APIs:
    - GitHub: Uses "token" prefix (auth_type="token")
    - GitLab: Uses "Bearer" prefix (auth_type="Bearer")
    - AWS: Uses SigV4 signing (auth_type="aws_sig_v4")

    Examples:
        # GitHub API with environment variable
        process_auth_headers({}, {"auth_type": "token", "auth_env_var": "GITHUB_TOKEN"})

        # GitLab API with environment variable
        process_auth_headers({}, {"auth_type": "Bearer", "auth_env_var": "GITLAB_TOKEN"})
    """
    headers = headers or {}

    # Get auth token from input or environment
    auth_token = tool_input.get("auth_token")
    if not auth_token and "auth_env_var" in tool_input:
        env_var_name = tool_input["auth_env_var"]
        auth_token = os.getenv(env_var_name)
        if not auth_token:
            raise ValueError(
                f"Environment variable '{env_var_name}' not found or empty. "
                f"Use environment(action='list') to see available variables."
            )

    auth_type = tool_input.get("auth_type")

    if auth_token:
        # Handle other auth types
        if auth_type == "Bearer":
            headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "token":
            # GitHub API uses 'token' prefix
            headers["Authorization"] = f"token {auth_token}"

            # Special case for GitHub API to add proper Accept header if not present
            if "Accept" not in headers and "github" in tool_input.get("url", "").lower():
                headers["Accept"] = "application/vnd.github.v3+json"

        elif auth_type == "custom":
            headers["Authorization"] = auth_token
        elif auth_type == "api_key":
            headers["X-API-Key"] = auth_token

    return headers


def stream_response(response: requests.Response) -> str:
    """Handle streaming response processing."""
    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks).decode()


def format_request_preview(method: str, url: str, headers: Dict, body: Optional[str] = None) -> Panel:
    """Format request details for preview."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Method", method)
    table.add_row("URL", url)

    # Add headers (hide sensitive values)
    headers_str = {}
    for key, value in headers.items():
        if key.lower() in ["authorization", "x-api-key", "cookie"]:
            # Show first 4 and last 4 chars if long enough, otherwise mask completely
            if isinstance(value, str) and len(value) > 12:
                headers_str[key] = f"{value[:4]}...{value[-4:]}"
            else:
                headers_str[key] = "********"
        else:
            headers_str[key] = value

    table.add_row("Headers", str(headers_str))

    if body:
        # Try to format body as JSON if it's valid
        try:
            json_body = json.loads(body)
            body = json.dumps(json_body, indent=2)
            body_preview = body[:200] + "..." if len(body) > 200 else body
            table.add_row("Body", f"(JSON) {body_preview}")
        except BaseException:
            body_preview = body[:200] + "..." if len(body) > 200 else body
            table.add_row("Body", body_preview)

    return Panel(
        table,
        title=f"[bold blue]🚀 HTTP Request Preview: {method} {urlparse(url).path}",
        border_style="blue",
        box=box.ROUNDED,
    )


def format_response_preview(
    response: requests.Response, content: str, metrics: Optional[Dict[Any, Any]] = None
) -> Panel:
    """Format response for preview."""
    status_code = response.status_code if response and hasattr(response, "status_code") else 0
    status_style = "green" if 200 <= status_code < 400 else "red"  # type: ignore

    # Main content panel
    main_table = Table(show_header=False, box=box.SIMPLE)
    main_table.add_column("Field", style="cyan")
    main_table.add_column("Value")

    # Status code with color
    main_table.add_row("Status", Text(f"{response.status_code} {response.reason}", style=status_style))

    # URL
    main_table.add_row("URL", response.url)

    # Content type
    content_type = response.headers.get("Content-Type", "unknown")
    main_table.add_row("Content-Type", content_type)

    # Size
    size_bytes = len(response.content)
    size_display = f"{size_bytes:,} bytes"
    if size_bytes > 1024:
        size_display += f" ({size_bytes / 1024:.1f} KB)"
    main_table.add_row("Size", size_display)

    # Timing if metrics available
    if metrics and "duration" in metrics:
        main_table.add_row("Duration", f"{metrics['duration']:.3f} seconds")

    # Format and preview content based on content type
    if "application/json" in content_type:
        try:
            # Format JSON for display
            json_obj = json.loads(content)
            # Create syntax highlighted JSON
            Syntax(
                json.dumps(json_obj, indent=2),
                "json",
                theme="monokai",
                line_numbers=False,
            )
        except BaseException:
            # Not valid JSON, show as text
            Text(content[:500] + "..." if len(content) > 500 else content)
    elif "text/html" in content_type:
        Syntax(
            content[:500] + "..." if len(content) > 500 else content,
            "html",
            theme="monokai",
            line_numbers=False,
        )
    else:
        # Default text preview
        Text(content[:500] + "..." if len(content) > 500 else content)

    # Combine into main panel
    status_emoji = "✅" if 200 <= status_code < 400 else "❌"  # type: ignore
    reason = response.reason if response and hasattr(response, "reason") else ""
    return Panel(
        Panel(main_table, border_style="blue", box=box.SIMPLE),
        title=f"[bold {status_style}]{status_emoji} HTTP Response: {status_code} {reason}",
        border_style=status_style,
        box=box.ROUNDED,
    )


def http_request(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Execute HTTP request with comprehensive authentication and features.

    Common API Examples:

    1. GitHub API (uses "token" auth_type):
        ```python
        http_request(
            method="GET",
            url="https://api.github.com/user",
            auth_type="token",
            auth_env_var="GITHUB_TOKEN",
        )
        ```

    2. GitLab API (uses "Bearer" auth_type):
        ```python
        http_request(
            method="GET",
            url="https://gitlab.com/api/v4/user",
            auth_type="Bearer",
            auth_env_var="GITLAB_TOKEN",
        )
        ```

    3. AWS S3 (uses "aws_sig_v4" auth_type):
        ```python
        http_request(
            method="GET",
            url="https://s3.amazonaws.com/my-bucket",
            auth_type="aws_sig_v4",
            aws_auth={"service": "s3"},
        )
        ```

    4. Using cookies from file and saving to cookie jar:
        ```python
        http_request(
            method="GET",
            url="https://internal-site.amazon.com",
            cookie="~/.midway/cookie",
            cookie_jar="~/.midway/cookie.updated",
        )
        ```

    5. Control redirect behavior:
        ```python
        http_request(
            method="GET",
            url="https://example.com/redirect",
            allow_redirects=True,  # Default behavior
            max_redirects=5,  # Limit number of redirects to follow
        )
        ```

    6. Convert HTML responses to markdown:
        ```python
        http_request(
            method="GET",
            url="https://example.com/article",
            convert_to_markdown=True,  # Converts HTML content to markdown
        )
        ```

    7. Using proxy:
        ```python
        http_request(
            method="GET",
            url="https://example.com/api",
            proxies={"https": "https://proxy.example.com:8080"},
        )
        ```

    Environment Variables:
    - Authentication tokens are read from environment when auth_env_var is specified
    - AWS credentials are automatically loaded from environment variables or credentials file
    - Use environment(action='list') to view all available environment variables
    """
    console = console_util.create()

    try:
        # Extract input from tool use object or use directly if already a dict
        tool_input = {}
        tool_use_id = "default_id"

        if isinstance(tool, dict):
            if "input" in tool:
                tool_input = tool["input"]
                tool_use_id = tool.get("toolUseId", "default_id")
            # No else here - tool_input has already been initialized

        method = tool_input["method"]
        url = tool_input["url"]
        headers = process_auth_headers(tool_input.get("headers", {}), tool_input)
        body = tool_input.get("body")
        verify = tool_input.get("verify_ssl", True)
        cookie = tool_input.get("cookie")
        cookie_jar = tool_input.get("cookie_jar")

        # Preview request before execution
        preview_panel = format_request_preview(method, url, headers, body)
        console.print(preview_panel)

        # Check if we're in development mode
        strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

        # For modifying operations (non-GET requests), show confirmation dialog unless in BYPASS_TOOL_CONSENT mode
        modifying_methods = {"POST", "PUT", "PATCH", "DELETE"}
        needs_confirmation = method.upper() in modifying_methods and not strands_dev

        if needs_confirmation:
            # Show warning for potentially modifying requests
            target_url = urlparse(url)
            warning_panel = Panel(
                Text.assemble(
                    ("⚠️ Warning: ", "bold red"),
                    (f"{method.upper()} request may modify data at ", "yellow"),
                    (f"{target_url.netloc}{target_url.path}", "bold yellow"),
                ),
                title="[bold red]Modifying Request Confirmation",
                border_style="red",
                box=box.DOUBLE,
                expand=False,
                padding=(1, 1),
            )
            console.print(warning_panel)

            # If body exists, show preview
            if body:
                try:
                    # Try to format as JSON
                    json_body = json.loads(body)
                    body_preview = json.dumps(json_body, indent=2)
                    console.print(
                        Panel(
                            Syntax(body_preview, "json", theme="monokai"),
                            title="[bold blue]Request Body Preview",
                            border_style="blue",
                            box=box.ROUNDED,
                        )
                    )
                except BaseException:
                    # Not JSON, show as plain text
                    console.print(
                        Panel(
                            Text(body[:500] + "..." if len(body) > 500 else body),
                            title="[bold blue]Request Body Preview",
                            border_style="blue",
                            box=box.ROUNDED,
                        )
                    )

            # Get user confirmation
            user_input = get_user_input(
                f"<yellow><bold>Do you want to proceed with this {method.upper()} request?</bold> [y/*]</yellow>"
            )
            if user_input.lower().strip() != "y":
                cancellation_reason = (
                    user_input
                    if user_input.strip() != "n"
                    else get_user_input("Please provide a reason for cancellation:")
                )
                error_message = f"HTTP request cancelled by the user. Reason: {cancellation_reason}"
                error_panel = Panel(
                    Text(error_message, style="bold red"),
                    title="[bold red]Request Cancelled",
                    border_style="red",
                    box=box.HEAVY,
                    expand=False,
                )
                console.print(error_panel)
                # Return error status for cancellation to ensure test passes
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": error_message}],
                }

        # Session handling
        session_config = tool_input.get("session_config", {})
        session = get_cached_session(url, session_config)

        # Authentication processing
        auth: Optional[Union[requests.auth.HTTPDigestAuth, AWSRequestsAuth]] = None
        if "auth_type" in tool_input:
            auth_type = tool_input["auth_type"]

            if auth_type == "digest":
                auth = handle_digest_auth(tool_input["digest_auth"], method, url)
            elif auth_type == "aws_sig_v4":
                auth = handle_aws_sigv4(tool_input["aws_auth"], url)
            elif auth_type == "basic":
                if "basic_auth" not in tool_input:
                    raise ValueError("basic_auth configuration required for basic authentication")
                basic_config = tool_input["basic_auth"]
                if "username" not in basic_config or "password" not in basic_config:
                    raise ValueError("username and password required for basic authentication")
                headers.update(handle_basic_auth(basic_config["username"], basic_config["password"]))
            elif auth_type == "jwt":
                headers.update(handle_jwt(tool_input["jwt_config"]))

        # Show request confirmation message
        console.print(Text("Sending request...", style="blue"))

        # Prepare request
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "verify": verify,
            "auth": auth,
            "allow_redirects": tool_input.get("allow_redirects", True),
            "proxies": tool_input.get("proxies", None),
        }

        # Set max_redirects if specified
        if "max_redirects" in tool_input:
            max_redirects = tool_input["max_redirects"]
            if max_redirects is not None and hasattr(session, "max_redirects"):
                session.max_redirects = max_redirects

        # Handle cookies
        if cookie:
            cookie_path = os.path.expanduser(cookie)
            if os.path.exists(cookie_path):
                cookies = http.cookiejar.MozillaCookieJar()
                try:
                    # Try Mozilla format first
                    cookies.load(cookie_path, ignore_discard=True, ignore_expires=True)
                    session.cookies.update(cookies)
                except Exception:
                    try:
                        # Try Netscape format (curl style)
                        with open(cookie_path, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    parts = line.split("\t")
                                    if len(parts) >= 7:  # Standard Netscape format
                                        (
                                            domain,
                                            flag,
                                            path,
                                            secure,
                                            expires,
                                            name,
                                            value,
                                        ) = parts
                                        session.cookies.set(name, value, domain=domain, path=path)
                    except Exception as e2:
                        console.print(
                            Text(
                                f"Failed to load cookies from {cookie}: {str(e2)}",
                                style="red",
                            )
                        )
                console.print(Text(f"Using cookies from {cookie}", style="blue"))
            else:
                console.print(Text(f"Warning: Cookie file {cookie} not found", style="yellow"))

        if body:
            request_kwargs["data"] = body

        # Execute request with metrics
        start_time = time.time()
        response = session.request(**request_kwargs)

        # Save cookies to cookie jar if specified
        if cookie_jar:
            cookie_jar_path = os.path.expanduser(cookie_jar)
            # Ensure directory exists
            cookie_jar_dir = os.path.dirname(cookie_jar_path)
            if cookie_jar_dir and not os.path.exists(cookie_jar_dir):
                os.makedirs(cookie_jar_dir, exist_ok=True)

            # Save cookies in Netscape format compatible with curl
            with open(cookie_jar_path, "w") as f:
                f.write("# Netscape HTTP Cookie File\n")
                f.write("# https://curl.se/docs/http-cookies.html\n")
                f.write("# This file was generated by Strands http_request tool\n\n")

                for cookie in session.cookies:
                    # Format is: domain flag path secure expires name value
                    secure = "TRUE" if cookie.secure else "FALSE"
                    httponly = "TRUE" if cookie.has_nonstandard_attr("httponly") else "FALSE"
                    expires = str(int(cookie.expires)) if hasattr(cookie, "expires") and cookie.expires else "0"
                    f.write(
                        f"{cookie.domain}\t{httponly}\t{cookie.path}\t{secure}\t{expires}\t{cookie.name}\t{cookie.value}\n"
                    )

            console.print(Text(f"Cookies saved to {cookie_jar}", style="blue"))

        # Process metrics if enabled
        metrics = None
        if tool_input.get("metrics", False):
            metrics = process_metrics(start_time, response)

        # Handle streaming responses
        if tool_input.get("streaming", False):
            content = stream_response(response)
        else:
            content = response.text

        # Convert HTML to markdown if requested
        convert_to_markdown = tool_input.get("convert_to_markdown", False)
        if convert_to_markdown:
            content_type = response.headers.get("content-type", "")
            is_html_content = (
                "text/html" in content_type.lower()
                or "<html" in content[:100].lower()
                or "<!doctype html" in content[:100].lower()
            )

            if is_html_content:
                original_content = content
                content = extract_content_from_html(content)

                # Add a note if conversion was successful
                if content != original_content:
                    console.print(Text("✓ Converted HTML content to markdown", style="green"))

        # Format and display the response
        response_panel = format_response_preview(response, content, metrics if metrics is not None else None)
        console.print(response_panel)

        # Show redirect information if redirects were followed
        if response.history:
            redirect_count = len(response.history)
            redirect_chain = " -> ".join([str(r.status_code) for r in response.history] + [str(response.status_code)])
            redirect_info = Panel(
                Text.assemble(
                    (f"Followed {redirect_count} redirect(s): ", "yellow"),
                    (redirect_chain, "cyan"),
                ),
                title="[bold blue]Redirect Information",
                border_style="blue",
                box=box.ROUNDED,
                expand=False,
                padding=(1, 1),
            )
            console.print(redirect_info)

        # Show headers in a separate table if response was successful
        if 200 <= response.status_code < 300:
            console.print(format_headers_table(dict(response.headers)))

        # Format response content for output
        result_text = []
        result_text.append(f"Status Code: {response.status_code}")

        # Add redirect information if any redirects were followed
        if response.history:
            redirect_count = len(response.history)
            redirect_chain = " -> ".join([str(r.status_code) for r in response.history] + [str(response.status_code)])
            result_text.append(f"Redirects: {redirect_count} redirects followed ({redirect_chain})")

        # Add minimal headers to text response
        important_headers = ["Content-Type", "Content-Length", "Date", "Server"]
        headers_text = {k: v for k, v in response.headers.items() if k in important_headers}
        result_text.append(f"Headers: {headers_text}")

        # Add body to text response
        result_text.append(f"Body: {content}")

        # Add metrics if available
        if metrics:
            result_text.append(f"Metrics: {metrics}")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": text} for text in result_text],
        }

    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="❌ HTTP Request Error",
            border_style="red",
            box=box.ROUNDED,
        )
        console.print(error_panel)

        # If the error appears to be related to authentication, show a suggestion
        error_str = str(e).lower()
        suggestion = ""
        if "auth" in error_str or "token" in error_str or "credential" in error_str or "unauthorized" in error_str:
            suggestion = (
                "\n\nSuggestion: Check your authentication setup. Common solutions:\n"
                "- For GitHub API: Use auth_type='token' with auth_env_var='GITHUB_TOKEN'\n"
                "- For GitLab API: Use auth_type='Bearer' with auth_env_var='GITLAB_TOKEN'\n"
                "- Use environment(action='list') to view available environment variables"
            )

        # Special handling for ImportError to help with test assertions
        error_text = f"Error: {str(e)}{suggestion}"
        if isinstance(e, ImportError):
            error_text = f"ImportError: {str(e)}{suggestion}"

        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_text}],
        }
