"""
Tests for the http_request tool using the Agent interface and direct invocation.
"""

import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import responses
from strands import Agent

from strands_tools import http_request


@pytest.fixture
def agent():
    """Create an agent with the http_request tool loaded."""
    return Agent(tools=[http_request])


@pytest.fixture
def mock_request_state():
    """Create a mock request state dictionary."""
    return {}


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    os.environ["TEST_TOKEN"] = "test-token-value"
    os.environ["GITHUB_TOKEN"] = "github-token-1234"
    os.environ["GITLAB_TOKEN"] = "gitlab-token-5678"
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return "\n".join([item["text"] for item in result["content"]])
    return str(result)


@responses.activate
def test_basic_get_request():
    """Test a basic GET request with direct invocation."""
    # Set up mock response
    responses.add(
        responses.GET,
        "https://example.com/api",
        json={"status": "success", "data": "test data"},
        status=200,
        content_type="application/json",
    )

    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"method": "GET", "url": "https://example.com/api"},
    }

    # Call the http_request function directly
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        # Mock user input so we don't wait for it
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"

    # Get all of the content text together
    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    # Check that the response body contains our mock data
    assert "Body:" in result_text
    assert "success" in result_text
    assert "test data" in result_text


@responses.activate
def test_post_request():
    """Test a POST request with JSON payload."""
    # Set up mock response
    responses.add(
        responses.POST,
        "https://example.com/api/create",
        json={"id": "123", "status": "created"},
        status=201,
        content_type="application/json",
    )

    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-post-id",
        "input": {
            "method": "POST",
            "url": "https://example.com/api/create",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"name": "Test Item", "value": 42}),
        },
    }

    # Call the http_request function directly
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"  # Confirm the POST request
        result = http_request.http_request(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"

    result_text = extract_result_text(result)
    assert "Status Code: 201" in result_text

    # Check that our request was sent correctly
    request = responses.calls[0].request
    assert request.method == "POST"
    assert json.loads(request.body) == {"name": "Test Item", "value": 42}


@responses.activate
def test_error_response():
    """Test handling of error responses."""
    # Set up mock error response
    responses.add(
        responses.GET,
        "https://example.com/api/error",
        json={"error": "Not found"},
        status=404,
    )

    tool_use = {
        "toolUseId": "test-error-id",
        "input": {"method": "GET", "url": "https://example.com/api/error"},
    }

    # Call the http_request function directly
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify the result shows the error status code
    assert result["status"] == "success"  # Even with HTTP 404, the tool itself should succeed

    result_text = extract_result_text(result)
    assert "Status Code: 404" in result_text


@responses.activate
def test_redirects():
    """Test following redirects."""
    # Set up redirect chain
    responses.add(
        responses.GET,
        "https://example.com/redirect",
        status=302,
        headers={"Location": "https://example.com/redirect2"},
    )

    responses.add(
        responses.GET,
        "https://example.com/redirect2",
        status=302,
        headers={"Location": "https://example.com/final"},
    )

    responses.add(
        responses.GET,
        "https://example.com/final",
        json={"status": "redirected"},
        status=200,
    )

    tool_use = {
        "toolUseId": "test-redirect-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/redirect",
            "allow_redirects": True,
        },
    }

    # Call the http_request function
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify redirect information is in the result
    assert result["status"] == "success"
    result_text = extract_result_text(result)
    assert "Redirects:" in result_text
    assert "redirected" in result_text  # Final response content


@responses.activate
def test_disable_redirects():
    """Test disabling redirects."""
    # Set up redirect
    responses.add(
        responses.GET,
        "https://example.com/redirect",
        status=302,
        headers={"Location": "https://example.com/final"},
    )

    tool_use = {
        "toolUseId": "test-no-redirect-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/redirect",
            "allow_redirects": False,
        },
    }

    # Call the http_request function
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify we get the 302 status code and no redirection
    assert result["status"] == "success"

    result_text = extract_result_text(result)
    assert "Status Code: 302" in result_text
    # Should not have "Redirects:" in the response
    assert "Redirects:" not in result_text


@responses.activate
def test_auth_token_direct(mock_env_vars):
    """Test using auth_token parameter directly."""
    responses.add(
        responses.GET,
        "https://api.example.com/protected",
        json={"status": "authenticated"},
        status=200,
        match=[responses.matchers.header_matcher({"Authorization": "Bearer test-token"})],
    )

    tool_use = {
        "toolUseId": "test-auth-id",
        "input": {
            "method": "GET",
            "url": "https://api.example.com/protected",
            "auth_type": "Bearer",
            "auth_token": "test-token",
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "success"
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-token"


@responses.activate
def test_auth_token_from_env(mock_env_vars):
    """Test getting auth token from environment variable."""
    responses.add(
        responses.GET,
        "https://api.example.com/protected",
        json={"status": "authenticated"},
        status=200,
        match=[responses.matchers.header_matcher({"Authorization": "Bearer test-token-value"})],
    )

    tool_use = {
        "toolUseId": "test-auth-env-id",
        "input": {
            "method": "GET",
            "url": "https://api.example.com/protected",
            "auth_type": "Bearer",
            "auth_env_var": "TEST_TOKEN",
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "success"
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-token-value"


@responses.activate
def test_github_api_auth(mock_env_vars):
    """Test GitHub API authentication with token prefix."""
    responses.add(
        responses.GET,
        "https://api.github.com/user",
        json={"login": "testuser"},
        status=200,
        match=[
            responses.matchers.header_matcher(
                {
                    "Authorization": "token github-token-1234",
                    "Accept": "application/vnd.github.v3+json",
                }
            )
        ],
    )

    tool_use = {
        "toolUseId": "test-github-id",
        "input": {
            "method": "GET",
            "url": "https://api.github.com/user",
            "auth_type": "token",
            "auth_env_var": "GITHUB_TOKEN",
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "success"
    assert len(responses.calls) == 1
    # Check that GitHub-specific headers were set
    assert responses.calls[0].request.headers["Authorization"] == "token github-token-1234"
    assert responses.calls[0].request.headers["Accept"] == "application/vnd.github.v3+json"


@responses.activate
def test_basic_auth():
    """Test basic authentication."""
    expected_header = "Basic " + "dXNlcjpwYXNzd29yZA=="  # user:password in base64

    responses.add(
        responses.GET,
        "https://api.example.com/basic-auth",
        json={"authenticated": True},
        status=200,
        match=[responses.matchers.header_matcher({"Authorization": expected_header})],
    )

    tool_use = {
        "toolUseId": "test-basic-auth-id",
        "input": {
            "method": "GET",
            "url": "https://api.example.com/basic-auth",
            "auth_type": "basic",
            "basic_auth": {"username": "user", "password": "password"},
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "success"
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["Authorization"] == expected_header


@responses.activate
def test_custom_headers():
    """Test sending custom headers."""
    responses.add(
        responses.GET,
        "https://api.example.com/custom-headers",
        json={"status": "success"},
        status=200,
        match=[responses.matchers.header_matcher({"X-Custom-Header": "custom-value", "User-Agent": "StrandsAgent"})],
    )

    tool_use = {
        "toolUseId": "test-headers-id",
        "input": {
            "method": "GET",
            "url": "https://api.example.com/custom-headers",
            "headers": {
                "X-Custom-Header": "custom-value",
                "User-Agent": "StrandsAgent",
            },
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "success"
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["X-Custom-Header"] == "custom-value"
    assert responses.calls[0].request.headers["User-Agent"] == "StrandsAgent"


@responses.activate
def test_cancellation(monkeypatch):
    """Test request cancellation by user."""
    # Temporarily override BYPASS_TOOL_CONSENT environment variable for this test
    original_env = os.environ.get("BYPASS_TOOL_CONSENT")
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "false")  # Force BYPASS_TOOL_CONSENT mode off

    try:
        # Register a mock response even though we'll cancel before sending
        responses.add(
            responses.POST,
            "https://example.com/api/create",
            json={"status": "success"},
            status=201,
        )

        tool_use = {
            "toolUseId": "test-cancel-id",
            "input": {
                "method": "POST",
                "url": "https://example.com/api/create",
                "body": json.dumps({"name": "Test Item"}),
            },
        }

        # Use a simplified approach - just mock the user input
        with patch("strands_tools.http_request.get_user_input") as mock_input:
            # First answer "n" to cancel, then provide a reason
            mock_input.side_effect = ["n", "Testing cancellation"]
            result = http_request.http_request(tool=tool_use)

        # Tool returns "error" status when the user cancels the operation
        assert result["status"] == "error"
        assert "HTTP request cancelled by the user" in result["content"][0]["text"]
        assert "Testing cancellation" in result["content"][0]["text"]
        # No request should have been sent
        assert len(responses.calls) == 0

    finally:
        # Restore original environment variable state
        if original_env is not None:
            monkeypatch.setenv("BYPASS_TOOL_CONSENT", original_env)
        else:
            monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)


@responses.activate
def test_missing_env_var():
    """Test error when environment variable doesn't exist."""
    tool_use = {
        "toolUseId": "test-missing-env-id",
        "input": {
            "method": "GET",
            "url": "https://api.example.com/",
            "auth_type": "Bearer",
            "auth_env_var": "NON_EXISTENT_TOKEN",
        },
    }

    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    assert result["status"] == "error"
    assert "Environment variable 'NON_EXISTENT_TOKEN' not found" in result["content"][0]["text"]


def test_aws_sigv4_auth():
    """Test AWS SigV4 authentication."""
    tool_use = {
        "toolUseId": "test-aws-auth-id",
        "input": {
            "method": "GET",
            "url": "https://s3.amazonaws.com/my-bucket",
            "auth_type": "aws_sig_v4",
            "aws_auth": {"service": "s3"},
        },
    }

    # Mock the AWS credential functions
    with patch("strands_tools.http_request.get_aws_credentials") as mock_creds:
        # Create mock credentials
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "test-access-key"
        mock_frozen_creds.secret_key = "test-secret-key"
        mock_frozen_creds.token = "test-session-token"

        # Return mock credentials and region
        mock_creds.return_value = (mock_frozen_creds, "us-west-2")

        # Mock AWSRequestsAuth
        with patch("strands_tools.http_request.AWSRequestsAuth") as mock_aws_auth:
            mock_auth = MagicMock()
            mock_aws_auth.return_value = mock_auth

            # Mock the request and get_user_input
            with (
                patch("requests.Session.request") as mock_request,
                patch("strands_tools.http_request.get_user_input") as mock_input,
            ):
                # Set up mock response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = '{"status": "success"}'
                mock_response.content = b'{"status": "success"}'
                mock_response.headers = {"Content-Type": "application/json"}
                mock_response.history = []
                mock_response.url = "https://s3.amazonaws.com/my-bucket"
                mock_response.request = MagicMock()
                mock_response.request.body = None
                mock_request.return_value = mock_response

                # Mock user input
                mock_input.return_value = "y"

                # Call the function
                result = http_request.http_request(tool=tool_use)

    # Verify AWS auth was called with right parameters
    mock_aws_auth.assert_called_once_with(
        aws_access_key="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_host="s3.amazonaws.com",
        aws_region="us-west-2",
        aws_service="s3",
        aws_token="test-session-token",
    )

    assert result["status"] == "success"

    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text


def test_jwt_auth():
    """Test JWT authentication."""
    tool_use = {
        "toolUseId": "test-jwt-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/protected",
            "auth_type": "jwt",
            "jwt_config": {
                "secret": "test-secret",
                "algorithm": "HS256",
                "expiry": 3600,
            },
        },
    }

    # Mock the handle_jwt function rather than the JWT module directly
    with (
        patch("strands_tools.http_request.handle_jwt") as mock_jwt_handler,
        patch("requests.Session.request") as mock_request,
        patch("strands_tools.http_request.get_user_input") as mock_input,
    ):
        # Configure mock JWT token handling
        mock_jwt_handler.return_value = {"Authorization": "Bearer mock-jwt-token"}

        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "authenticated"}'
        mock_response.content = b'{"status": "authenticated"}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.history = []
        mock_response.url = "https://example.com/api/protected"
        mock_response.request = MagicMock()
        mock_response.request.body = None
        mock_request.return_value = mock_response

        # Mock user input
        mock_input.return_value = "y"

        # Call function
        result = http_request.http_request(tool=tool_use)

    # Verify JWT handler was called
    mock_jwt_handler.assert_called_once()

    # Verify request was made with JWT token
    assert mock_request.call_args is not None
    headers = mock_request.call_args[1]["headers"]
    assert headers.get("Authorization") == "Bearer mock-jwt-token"

    # Verify result
    assert result["status"] == "success"


@responses.activate
def test_verify_ssl_option():
    """Test the verify_ssl option."""

    # Set up mock response
    responses.add(
        responses.GET,
        "https://example.com/api/insecure",
        json={"status": "insecure"},
        status=200,
    )

    tool_use = {
        "toolUseId": "test-ssl-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/insecure",
            "verify_ssl": False,
        },
    }

    # Call http_request with verify_ssl=False
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        # Use a real request but don't actually send it over the network
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://example.com/api/insecure",
                json={"status": "insecure"},
                status=200,
            )
            result = http_request.http_request(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"


@responses.activate
def test_dev_mode_no_confirmation():
    """Test that in BYPASS_TOOL_CONSENT mode, no confirmation is requested for modifying requests."""
    # Set up mock POST response
    responses.add(
        responses.POST,
        "https://example.com/api/create-no-confirm",
        json={"status": "created without confirmation"},
        status=201,
    )

    tool_use = {
        "toolUseId": "test-dev-mode-id",
        "input": {
            "method": "POST",
            "url": "https://example.com/api/create-no-confirm",
            "body": json.dumps({"name": "Test Item"}),
        },
    }

    # Set BYPASS_TOOL_CONSENT environment variable
    original_env = os.environ.copy()
    os.environ["BYPASS_TOOL_CONSENT"] = "true"

    try:
        # In BYPASS_TOOL_CONSENT mode, get_user_input should not be called for confirmation
        with patch("strands_tools.http_request.get_user_input") as mock_input:
            # This will be called if the test fails
            mock_input.side_effect = AssertionError("Should not ask for confirmation in BYPASS_TOOL_CONSENT mode")
            result = http_request.http_request(tool=tool_use)

        # Verify the result
        assert result["status"] == "success"
        assert len(responses.calls) == 1
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@responses.activate
def test_cookie_handling(tmp_path):
    """Test loading cookies from a file and saving to cookie jar."""
    # Create a temporary cookie file
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        """# Netscape HTTP Cookie File
# https://curl.se/docs/http-cookies.html
example.com	FALSE	/	FALSE	0	session_id	abc123
example.com	FALSE	/	FALSE	0	user_pref	dark_mode
"""
    )

    # Create a cookie jar path for output
    cookie_jar_path = tmp_path / "cookies_out.txt"

    # Set up mock response with cookies
    responses.add(
        responses.GET,
        "https://example.com/api/cookies",
        json={"status": "with cookies"},
        status=200,
    )

    tool_use = {
        "toolUseId": "test-cookie-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/cookies",
            "cookie": str(cookie_file),
            "cookie_jar": str(cookie_jar_path),
        },
    }

    # Call http_request with cookie handling
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"

    # Verify cookie jar was created
    assert cookie_jar_path.exists()
    cookie_jar_content = cookie_jar_path.read_text()
    assert "Netscape HTTP Cookie File" in cookie_jar_content

    # Check that at least one request was made
    assert len(responses.calls) >= 1


@responses.activate
def test_metrics_collection_with_detail():
    """Test metrics collection functionality."""
    responses.add(
        responses.GET,
        "https://example.com/api/metrics",
        json={"status": "metrics test"},
        status=200,
    )

    tool_use = {
        "toolUseId": "test-metrics-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/metrics",
            "metrics": True,
        },
    }

    # Clear any existing metrics
    if hasattr(http_request, "REQUEST_METRICS"):
        http_request.REQUEST_METRICS.clear()

    # Call http_request with metrics enabled
    with (
        patch("strands_tools.http_request.get_user_input") as mock_input,
        patch(
            "strands_tools.http_request.process_metrics",
            wraps=http_request.process_metrics,
        ) as mock_metrics,
    ):
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify metrics were processed
    assert mock_metrics.called

    # Verify metrics were stored in REQUEST_METRICS
    assert len(http_request.REQUEST_METRICS) > 0
    assert urlparse("example.com").netloc in [urlparse(url).netloc for url in http_request.REQUEST_METRICS]
    assert len(http_request.REQUEST_METRICS["example.com"]) > 0

    # Verify metrics data structure
    metrics_entry = http_request.REQUEST_METRICS["example.com"][0]
    assert "duration" in metrics_entry
    assert "status_code" in metrics_entry
    assert metrics_entry["status_code"] == 200

    # Verify metrics are mentioned in the result
    result_text = extract_result_text(result)
    assert "Metrics:" in result_text


@responses.activate
def test_session_reuse():
    """Test session reuse for multiple requests to the same domain."""
    # Add two responses for the same domain
    responses.add(
        responses.GET,
        "https://example.com/api/session1",
        json={"session": "first"},
        status=200,
    )

    responses.add(
        responses.GET,
        "https://example.com/api/session2",
        json={"session": "second"},
        status=200,
    )

    # Clear session cache
    http_request.SESSION_CACHE.clear()

    # Make first request
    tool_use1 = {
        "toolUseId": "test-session1-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/session1",
        },
    }

    # Make second request
    tool_use2 = {
        "toolUseId": "test-session2-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/session2",
        },
    }

    # Patch get_cached_session to track calls
    with (
        patch(
            "strands_tools.http_request.get_cached_session",
            wraps=http_request.get_cached_session,
        ) as mock_session,
        patch("strands_tools.http_request.get_user_input") as mock_input,
    ):
        mock_input.return_value = "y"

        # Make first request
        result1 = http_request.http_request(tool=tool_use1)

        # Make second request
        result2 = http_request.http_request(tool=tool_use2)

    # Verify both requests succeeded
    assert result1["status"] == "success"
    assert result2["status"] == "success"

    # Verify session was reused (get_cached_session should return cached session for second request)
    assert len(http_request.SESSION_CACHE) == 1
    assert urlparse("example.com").netloc in [urlparse(url).netloc for url in http_request.SESSION_CACHE]
    assert mock_session.call_count == 2


@responses.activate
def test_invalid_jwt_import():
    """Test error handling when PyJWT is not installed."""
    tool_use = {
        "toolUseId": "test-jwt-import-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/jwt",
            "auth_type": "jwt",
            "jwt_config": {
                "secret": "test-secret",
                "algorithm": "HS256",
                "expiry": 3600,
            },
        },
    }

    # Mock handle_jwt to simulate import error
    with (
        patch("strands_tools.http_request.handle_jwt") as mock_jwt_handler,
        patch("strands_tools.http_request.get_user_input") as mock_input,
    ):
        # Make jwt import raise ImportError
        mock_jwt_handler.side_effect = ImportError("No module named 'jwt'")
        mock_input.return_value = "y"

        # Call function and verify it handles the error
        result = http_request.http_request(tool=tool_use)

    # Verify error response
    assert result["status"] == "error"
    assert "ImportError" in result["content"][0]["text"]
    assert "jwt" in result["content"][0]["text"].lower()

    # Remove the metrics check as it might not be included when an import error occurs
    # The function exits early with an error, before metrics are collected


@responses.activate
def test_streaming_response():
    """Test handling of streaming responses."""
    # Set up mock response
    responses.add(
        responses.GET,
        "https://example.com/api/stream",
        body="chunk1chunk2chunk3",
        status=200,
        stream=True,
    )

    tool_use = {
        "toolUseId": "test-stream-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/stream",
            "streaming": True,
        },
    }

    # Patch the stream_response function to verify it's called
    with (
        patch("strands_tools.http_request.get_user_input") as mock_input,
        patch(
            "strands_tools.http_request.stream_response",
            wraps=http_request.stream_response,
        ) as mock_stream,
    ):
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"
    assert mock_stream.called

    # Check output contains expected content
    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    assert "chunk1chunk2chunk3" in result_text


@responses.activate
def test_cookie_file_not_found():
    """Test handling when cookie file doesn't exist."""
    responses.add(
        responses.GET,
        "https://example.com/api/cookies-missing",
        json={"status": "no cookies"},
        status=200,
    )

    tool_use = {
        "toolUseId": "test-missing-cookie-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/cookies-missing",
            "cookie": "/nonexistent/cookie/file.txt",
        },
    }

    # Call http_request with non-existent cookie file
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(tool=tool_use)

    # Verify the request still succeeds
    assert result["status"] == "success"

    # Check that a warning message was printed (would need to capture console output)
    # This is hard to test directly, but the function should not fail


@responses.activate
def test_http_request_via_agent(agent):
    """Test HTTP request via Agent interface."""
    # Set up mock response
    responses.add(
        responses.GET,
        "https://example.com/api/agent-test",
        json={"status": "success via agent"},
        status=200,
    )

    # Test with agent interface
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = agent.tool.http_request(method="GET", url="https://example.com/api/agent-test")

    # Extract and check result
    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    assert "success via agent" in result_text


@responses.activate
def test_markdown_conversion():
    """Test HTML to markdown conversion functionality."""
    # Mock HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is a paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
        <a href="https://example.com">Link to example</a>
    </body>
    </html>
    """

    # Set up mock response with HTML content
    responses.add(responses.GET, "https://example.com/article", body=html_content, status=200, content_type="text/html")

    # Test without markdown conversion (should return HTML)
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(
            {"input": {"method": "GET", "url": "https://example.com/article"}, "toolUseId": "test1"}
        )

    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    assert "<html>" in result_text  # Should contain HTML

    # Test with markdown conversion (should convert to markdown if packages available)
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(
            {
                "input": {"method": "GET", "url": "https://example.com/article", "convert_to_markdown": True},
                "toolUseId": "test2",
            }
        )

    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    # Verify markdown conversion worked - HTML tags should be removed and text content preserved
    assert "<html>" not in result_text  # HTML tags should be gone
    assert "<h1>" not in result_text
    assert "Main Heading" in result_text  # Text content should remain
    assert "bold text" in result_text
    assert "italic text" in result_text
    assert "List item 1" in result_text
    assert "List item 2" in result_text


@responses.activate
def test_markdown_conversion_non_html():
    """Test that non-HTML content is not affected by markdown conversion."""
    # Set up mock response with JSON content
    responses.add(
        responses.GET,
        "https://example.com/api/data",
        json={"message": "hello", "data": [1, 2, 3]},
        status=200,
    )

    # Test with markdown conversion enabled on non-HTML content
    with patch("strands_tools.http_request.get_user_input") as mock_input:
        mock_input.return_value = "y"
        result = http_request.http_request(
            {
                "input": {"method": "GET", "url": "https://example.com/api/data", "convert_to_markdown": True},
                "toolUseId": "test3",
            }
        )

    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
    assert '"message": "hello"' in result_text  # Should still be JSON (no conversion for non-HTML)


def test_proxy_support():
    """Test HTTP proxy support functionality."""
    tool_use = {
        "toolUseId": "test-proxy-id",
        "input": {
            "method": "GET",
            "url": "https://example.com/api/proxy-test",
            "proxies": {"https": "https://proxy.example.com:8080"},
        },
    }

    # Mock the session.request method to capture the proxies parameter
    with (
        patch("strands_tools.http_request.get_user_input") as mock_input,
        patch("requests.Session.request") as mock_request,
    ):
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "success via proxy"}'
        mock_response.content = b'{"status": "success via proxy"}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.history = []
        mock_response.url = "https://example.com/api/proxy-test"
        mock_response.request = MagicMock()
        mock_response.request.body = None
        mock_request.return_value = mock_response

        # Mock user input
        mock_input.return_value = "y"

        # Call the function
        result = http_request.http_request(tool=tool_use)

    # Verify the proxy was actually passed to requests
    assert mock_request.called
    call_kwargs = mock_request.call_args[1]
    assert "proxies" in call_kwargs
    assert call_kwargs["proxies"] == {"https": "https://proxy.example.com:8080"}

    # Verify the result
    assert result["status"] == "success"
    result_text = extract_result_text(result)
    assert "Status Code: 200" in result_text
