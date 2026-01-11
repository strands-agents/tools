"""
Tests for AWS CLI Documentation tools.

These tests use mocked HTTP responses to avoid hitting the real AWS documentation site.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from strands_tools.aws_cli_docs import (
    AwsCliDocsCache,
    _fetch_page,
    _parse_command_details,
    _parse_service_commands,
    get_aws_command_details,
    get_aws_service_commands,
)


# Sample HTML responses for mocking
SAMPLE_SERVICE_INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>S3 Commands</title></head>
<body>
<div class="toctree-wrapper">
    <ul>
        <li class="toctree-l1"><a href="cp.html">cp</a></li>
        <li class="toctree-l1"><a href="ls.html">ls</a></li>
        <li class="toctree-l1"><a href="mb.html">mb</a></li>
        <li class="toctree-l1"><a href="mv.html">mv</a></li>
        <li class="toctree-l1"><a href="rb.html">rb</a></li>
        <li class="toctree-l1"><a href="rm.html">rm</a></li>
        <li class="toctree-l1"><a href="sync.html">sync</a></li>
    </ul>
</div>
</body>
</html>
"""

SAMPLE_COMMAND_DETAILS_HTML = """
<!DOCTYPE html>
<html>
<head><title>aws s3 cp</title></head>
<body>
<div id="main-content">
    <h1>cp</h1>
    
    <div id="description">
        <h2>Description</h2>
        <p>Copies a local file or S3 object to another location locally or in S3.</p>
        <p>The cp command uses the following syntax.</p>
    </div>
    
    <div id="synopsis">
        <h2>Synopsis</h2>
        <pre>aws s3 cp &lt;LocalPath&gt; &lt;S3Uri&gt; or &lt;S3Uri&gt; &lt;LocalPath&gt; or &lt;S3Uri&gt; &lt;S3Uri&gt;
[--dryrun]
[--quiet]
[--include &lt;value&gt;]
[--exclude &lt;value&gt;]
[--acl &lt;value&gt;]
[--follow-symlinks | --no-follow-symlinks]
[--no-guess-mime-type]
[--sse &lt;value&gt;]
[--sse-c &lt;value&gt;]
[--recursive]</pre>
    </div>
    
    <div id="options">
        <h2>Options</h2>
        <dl>
            <dt>--dryrun</dt>
            <dd>Displays the operations that would be performed using the specified command without actually running them.</dd>
            <dt>--quiet</dt>
            <dd>Does not display the operations performed from the specified command.</dd>
            <dt>--recursive</dt>
            <dd>Command is performed on all files or objects under the specified directory or prefix.</dd>
        </dl>
    </div>
    
    <div id="examples">
        <h2>Examples</h2>
        <p>The following cp command copies a single file to a bucket:</p>
        <pre>aws s3 cp test.txt s3://mybucket/test2.txt</pre>
        <p>The following cp command copies a single file from S3 to local:</p>
        <pre>aws s3 cp s3://mybucket/test.txt test2.txt</pre>
    </div>
</div>
</body>
</html>
"""

SAMPLE_ALTERNATIVE_INDEX_HTML = """
<!DOCTYPE html>
<html>
<body>
<ul>
    <li><a href="create-function.html">create-function</a></li>
    <li><a href="invoke.html">invoke</a></li>
    <li><a href="list-functions.html">list-functions</a></li>
    <li><a href="index.html">index</a></li>
</ul>
</body>
</html>
"""


class TestAwsCliDocsCache:
    """Tests for the cache implementation."""

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = AwsCliDocsCache(ttl=3600)
        cache.set("test_key", {"data": "test_value"})

        result = cache.get("test_key")
        assert result == {"data": "test_value"}

    def test_cache_miss(self):
        """Test cache returns None for missing keys."""
        cache = AwsCliDocsCache(ttl=3600)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        cache = AwsCliDocsCache(ttl=0)  # Immediate expiration
        cache.set("test_key", {"data": "test_value"})

        # Entry should be expired immediately
        result = cache.get("test_key")
        assert result is None

    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        cache = AwsCliDocsCache(ttl=3600)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestParseServiceCommands:
    """Tests for parsing service command pages."""

    def test_parse_toctree_format(self):
        """Test parsing commands from toctree format."""
        commands = _parse_service_commands(SAMPLE_SERVICE_INDEX_HTML, "s3")

        assert len(commands) == 7
        assert {
            "command": "cp",
            "link": "https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/cp.html",
        } in commands
        assert {
            "command": "ls",
            "link": "https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/ls.html",
        } in commands

    def test_parse_alternative_format(self):
        """Test parsing commands from alternative HTML format."""
        commands = _parse_service_commands(SAMPLE_ALTERNATIVE_INDEX_HTML, "lambda")

        assert len(commands) >= 3
        command_names = [c["command"] for c in commands]
        assert "create-function" in command_names
        assert "invoke" in command_names
        assert "index" not in command_names  # Should exclude index

    def test_parse_empty_page(self):
        """Test parsing returns empty list for page with no commands."""
        commands = _parse_service_commands("<html><body></body></html>", "empty")
        assert commands == []


class TestParseCommandDetails:
    """Tests for parsing command details pages."""

    def test_parse_full_documentation(self):
        """Test parsing a complete command documentation page."""
        details = _parse_command_details(SAMPLE_COMMAND_DETAILS_HTML)

        assert "description" in details
        assert "Copies a local file" in details["description"]

        assert "synopsis" in details
        assert "aws s3 cp" in details["synopsis"]

        assert "options" in details
        assert "--dryrun" in details["options"]
        assert "--recursive" in details["options"]

        assert "examples" in details
        assert "aws s3 cp test.txt" in details["examples"]

    def test_parse_minimal_page(self):
        """Test parsing a minimal page with limited content."""
        minimal_html = "<html><body><p>Some documentation content</p></body></html>"
        details = _parse_command_details(minimal_html)

        # Should fall back to full_text extraction
        assert "full_text" in details or len(details) == 0


class TestGetAwsServiceCommands:
    """Tests for the get_aws_service_commands tool."""

    @patch("strands_tools.aws_cli_docs._fetch_page")
    @patch("strands_tools.aws_cli_docs._cache")
    def test_successful_fetch(self, mock_cache, mock_fetch):
        """Test successful command list retrieval."""
        mock_cache.get.return_value = None
        mock_fetch.return_value = SAMPLE_SERVICE_INDEX_HTML

        result = get_aws_service_commands(service="s3", use_cache=False)

        # Should return JSON with commands
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        assert "command" in parsed[0]
        assert "link" in parsed[0]

    @patch("strands_tools.aws_cli_docs._fetch_page")
    @patch("strands_tools.aws_cli_docs._cache")
    def test_service_not_found(self, mock_cache, mock_fetch):
        """Test handling of non-existent service."""
        mock_cache.get.return_value = None
        mock_fetch.return_value = None  # Simulates 404

        result = get_aws_service_commands(service="nonexistent", use_cache=False)

        parsed = json.loads(result)
        assert "error" in parsed

    @patch("strands_tools.aws_cli_docs._cache")
    def test_uses_cache(self, mock_cache):
        """Test that cached results are used when available."""
        cached_data = [{"command": "cached-cmd", "link": "https://cached"}]
        mock_cache.get.return_value = cached_data

        result = get_aws_service_commands(service="s3", use_cache=True)

        parsed = json.loads(result)
        assert parsed == cached_data

    @patch("strands_tools.aws_cli_docs._fetch_page")
    @patch("strands_tools.aws_cli_docs._cache")
    def test_service_name_normalization(self, mock_cache, mock_fetch):
        """Test that service names are normalized (lowercase, trimmed)."""
        mock_cache.get.return_value = None
        mock_fetch.return_value = SAMPLE_SERVICE_INDEX_HTML

        # Should normalize "S3" to "s3"
        result = get_aws_service_commands(service="  S3  ", use_cache=False)

        parsed = json.loads(result)
        assert isinstance(parsed, list)


class TestGetAwsCommandDetails:
    """Tests for the get_aws_command_details tool."""

    @patch("strands_tools.aws_cli_docs._fetch_page")
    @patch("strands_tools.aws_cli_docs._cache")
    def test_successful_fetch(self, mock_cache, mock_fetch):
        """Test successful command details retrieval."""
        mock_cache.get.return_value = None
        mock_fetch.return_value = SAMPLE_COMMAND_DETAILS_HTML

        result = get_aws_command_details(service="s3", command="cp", use_cache=False)

        parsed = json.loads(result)
        assert parsed["service"] == "s3"
        assert parsed["command"] == "cp"
        assert "link" in parsed
        assert "details" in parsed
        assert "synopsis" in parsed["details"]

    @patch("strands_tools.aws_cli_docs._fetch_page")
    @patch("strands_tools.aws_cli_docs._cache")
    def test_command_not_found(self, mock_cache, mock_fetch):
        """Test handling of non-existent command."""
        mock_cache.get.return_value = None
        mock_fetch.return_value = None  # Simulates 404

        result = get_aws_command_details(service="s3", command="nonexistent", use_cache=False)

        parsed = json.loads(result)
        assert "error" in parsed

    @patch("strands_tools.aws_cli_docs._cache")
    def test_uses_cache(self, mock_cache):
        """Test that cached results are used when available."""
        cached_data = {
            "service": "s3",
            "command": "cp",
            "link": "https://cached",
            "details": {"synopsis": "cached synopsis"},
        }
        mock_cache.get.return_value = cached_data

        result = get_aws_command_details(service="s3", command="cp", use_cache=True)

        parsed = json.loads(result)
        assert parsed == cached_data


class TestFetchPage:
    """Tests for the HTTP fetch function."""

    @patch("strands_tools.aws_cli_docs.requests.get")
    def test_successful_fetch(self, mock_get):
        """Test successful page fetch."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = _fetch_page("https://example.com")

        assert result == "<html>content</html>"

    @patch("strands_tools.aws_cli_docs.requests.get")
    def test_handles_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_get.side_effect = requests.exceptions.HTTPError("Not Found")

        result = _fetch_page("https://example.com/notfound")

        assert result is None

    @patch("strands_tools.aws_cli_docs.requests.get")
    def test_handles_connection_error(self, mock_get):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = _fetch_page("https://example.com")

        assert result is None


# Integration-style test (requires network, skip in CI)
@pytest.mark.skip(reason="Integration test - requires network access")
class TestIntegration:
    """Integration tests that hit the real AWS documentation."""

    def test_fetch_real_s3_commands(self):
        """Test fetching real S3 commands from AWS docs."""
        result = get_aws_service_commands(service="s3", use_cache=False)
        parsed = json.loads(result)

        assert len(parsed) > 5
        command_names = [c["command"] for c in parsed]
        assert "cp" in command_names
        assert "ls" in command_names

    def test_fetch_real_s3_cp_details(self):
        """Test fetching real S3 cp command details."""
        result = get_aws_command_details(service="s3", command="cp", use_cache=False)
        parsed = json.loads(result)

        assert parsed["service"] == "s3"
        assert parsed["command"] == "cp"
        assert "details" in parsed
