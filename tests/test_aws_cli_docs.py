"""Tests for AWS CLI documentation tools."""

from unittest.mock import MagicMock, patch

import pytest

from strands_tools.aws_cli_docs import (
    _fetch_page,
    _get_cached,
    _parse_command_details,
    _parse_service_commands,
    _set_cached,
    clear_cache,
    get_aws_command_details,
    get_aws_service_commands,
)

# Sample HTML for S3 service index page
SAMPLE_S3_INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>s3 - AWS CLI Command Reference</title></head>
<body>
<h1>s3</h1>
<div class="toctree-wrapper compound">
<ul>
<li class="toctree-l1"><a class="reference internal" href="cp.html">cp</a></li>
<li class="toctree-l1"><a class="reference internal" href="ls.html">ls</a></li>
<li class="toctree-l1"><a class="reference internal" href="mb.html">mb</a></li>
<li class="toctree-l1"><a class="reference internal" href="mv.html">mv</a></li>
<li class="toctree-l1"><a class="reference internal" href="rm.html">rm</a></li>
<li class="toctree-l1"><a class="reference internal" href="sync.html">sync</a></li>
</ul>
</div>
</body>
</html>
"""

# Sample HTML for S3 cp command page
SAMPLE_S3_CP_HTML = """
<!DOCTYPE html>
<html>
<head><title>cp - AWS CLI Command Reference</title></head>
<body>
<h1>cp</h1>
<section id="synopsis">
<h2>Synopsis</h2>
<pre>aws s3 cp &lt;LocalPath&gt; &lt;S3Uri&gt; [--options]</pre>
</section>
<section id="description">
<h2>Description</h2>
<p>Copies a local file or S3 object to another location locally or in S3.</p>
<p>The cp command supports various options for copying files.</p>
</section>
<section id="options">
<h2>Options</h2>
<dl>
<dt>--recursive</dt>
<dd>Command is performed on all files or objects under the specified directory or prefix.</dd>
<dt>--exclude</dt>
<dd>Exclude all files or objects from the command that matches the specified pattern.</dd>
<dt>--include</dt>
<dd>Include only files or objects that match the specified pattern.</dd>
</dl>
</section>
<section id="examples">
<h2>Examples</h2>
<pre>aws s3 cp test.txt s3://mybucket/test2.txt</pre>
<pre>aws s3 cp s3://mybucket/test.txt ./</pre>
</section>
</body>
</html>
"""


class TestCaching:
    """Tests for cache functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_set_and_get_cached(self):
        """Test setting and getting cached values."""
        _set_cached("test_key", {"data": "test"})
        result = _get_cached("test_key")
        assert result == {"data": "test"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = _get_cached("nonexistent_key")
        assert result is None

    def test_cache_expiry(self):
        """Test that expired cache entries are not returned."""
        import time

        from strands_tools import aws_cli_docs

        # Set a very short TTL
        original_ttl = aws_cli_docs._cache_ttl_seconds
        aws_cli_docs._cache_ttl_seconds = 0.01  # 10ms

        _set_cached("expiring_key", {"data": "test"})
        time.sleep(0.02)  # Wait for expiry

        result = _get_cached("expiring_key")
        assert result is None

        # Restore original TTL
        aws_cli_docs._cache_ttl_seconds = original_ttl

    def test_clear_cache(self):
        """Test clearing the cache."""
        _set_cached("key1", "value1")
        _set_cached("key2", "value2")

        clear_cache()

        assert _get_cached("key1") is None
        assert _get_cached("key2") is None


class TestParseServiceCommands:
    """Tests for parsing service index pages."""

    def test_parse_s3_commands(self):
        """Test parsing S3 service commands from HTML."""
        commands = _parse_service_commands(SAMPLE_S3_INDEX_HTML, "s3")

        assert len(commands) == 6
        command_names = [cmd["command"] for cmd in commands]
        assert "cp" in command_names
        assert "ls" in command_names
        assert "sync" in command_names

    def test_parse_empty_page(self):
        """Test parsing an empty HTML page."""
        commands = _parse_service_commands("<html><body></body></html>", "test")
        assert commands == []

    def test_command_links_are_valid(self):
        """Test that command links are properly constructed."""
        commands = _parse_service_commands(SAMPLE_S3_INDEX_HTML, "s3")

        for cmd in commands:
            assert "link" in cmd
            assert cmd["link"].startswith("https://awscli.amazonaws.com/")
            assert cmd["link"].endswith(".html")


class TestParseCommandDetails:
    """Tests for parsing command documentation pages."""

    def test_parse_cp_details(self):
        """Test parsing S3 cp command details."""
        details = _parse_command_details(SAMPLE_S3_CP_HTML)

        assert "synopsis" in details
        assert "aws s3 cp" in details["synopsis"]

        assert "description" in details
        assert "Copies" in details["description"]

        assert "options" in details
        assert len(details["options"]) == 3

        assert "examples" in details
        assert len(details["examples"]) == 2

    def test_parse_empty_page(self):
        """Test parsing an empty HTML page returns empty dict."""
        details = _parse_command_details("<html><body></body></html>")
        assert details == {} or not details.get("synopsis")

    def test_parse_title(self):
        """Test that title is extracted."""
        details = _parse_command_details(SAMPLE_S3_CP_HTML)
        assert "title" in details
        assert details["title"] == "cp"


class TestFetchPage:
    """Tests for HTTP fetching."""

    @patch("strands_tools.aws_cli_docs.httpx.Client")
    def test_fetch_page_success(self, mock_client_class):
        """Test successful page fetch."""
        mock_response = MagicMock()
        mock_response.text = "<html>test</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = _fetch_page("https://example.com")
        assert result == "<html>test</html>"

    @patch("strands_tools.aws_cli_docs.httpx.Client")
    def test_fetch_page_404(self, mock_client_class):
        """Test 404 response returns None."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = _fetch_page("https://example.com/notfound")
        assert result is None


class TestGetAwsServiceCommands:
    """Tests for get_aws_service_commands tool."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_s3_commands(self, mock_fetch):
        """Test getting S3 service commands."""
        mock_fetch.return_value = SAMPLE_S3_INDEX_HTML

        result = get_aws_service_commands(service="s3", use_cache=False)

        assert "s3" in result
        assert "cp" in result
        assert "ls" in result
        assert "6 commands" in result

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_commands_cached(self, mock_fetch):
        """Test that commands are cached."""
        mock_fetch.return_value = SAMPLE_S3_INDEX_HTML

        # First call - should fetch
        get_aws_service_commands(service="s3", use_cache=True)
        assert mock_fetch.call_count == 1

        # Second call - should use cache
        result = get_aws_service_commands(service="s3", use_cache=True)
        assert mock_fetch.call_count == 1
        assert "cached" in result.lower() or "6 commands" in result

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_commands_invalid_service(self, mock_fetch):
        """Test getting commands for invalid service."""
        mock_fetch.return_value = None

        result = get_aws_service_commands(service="invalidservice")

        assert "Could not find" in result or "error" in result.lower()

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_service_name_normalized(self, mock_fetch):
        """Test that service names are normalized."""
        mock_fetch.return_value = SAMPLE_S3_INDEX_HTML

        get_aws_service_commands(service="  S3  ", use_cache=False)

        # Should have called with normalized URL
        mock_fetch.assert_called_once()
        call_url = mock_fetch.call_args[0][0]
        assert "/s3/index.html" in call_url


class TestGetAwsCommandDetails:
    """Tests for get_aws_command_details tool."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_cp_details(self, mock_fetch):
        """Test getting S3 cp command details."""
        mock_fetch.return_value = SAMPLE_S3_CP_HTML

        result = get_aws_command_details(service="s3", command="cp", use_cache=False)

        assert "s3" in result
        assert "cp" in result
        assert "Synopsis" in result
        assert "Description" in result

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_details_cached(self, mock_fetch):
        """Test that command details are cached."""
        mock_fetch.return_value = SAMPLE_S3_CP_HTML

        # First call - should fetch
        get_aws_command_details(service="s3", command="cp", use_cache=True)
        assert mock_fetch.call_count == 1

        # Second call - should use cache
        get_aws_command_details(service="s3", command="cp", use_cache=True)
        assert mock_fetch.call_count == 1

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_get_details_invalid_command(self, mock_fetch):
        """Test getting details for invalid command."""
        mock_fetch.return_value = None

        result = get_aws_command_details(service="s3", command="invalidcmd")

        assert "Could not find" in result or "error" in result.lower()

    @patch("strands_tools.aws_cli_docs._fetch_page")
    def test_command_name_normalized(self, mock_fetch):
        """Test that command names are normalized."""
        mock_fetch.return_value = SAMPLE_S3_CP_HTML

        get_aws_command_details(service="S3", command="  CP  ", use_cache=False)

        # Should have called with normalized URL
        mock_fetch.assert_called_once()
        call_url = mock_fetch.call_args[0][0]
        assert "/s3/cp.html" in call_url


class TestIntegration:
    """Integration tests (require network access)."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @pytest.mark.integration
    def test_real_s3_commands(self):
        """Test fetching real S3 commands (requires network)."""
        result = get_aws_service_commands(service="s3", use_cache=False)

        assert "cp" in result
        assert "ls" in result
        assert "sync" in result

    @pytest.mark.integration
    def test_real_s3_cp_details(self):
        """Test fetching real S3 cp details (requires network)."""
        result = get_aws_command_details(service="s3", command="cp", use_cache=False)

        assert "Synopsis" in result
        assert "recursive" in result.lower() or "--recursive" in result

    @pytest.mark.integration
    def test_real_ec2_commands(self):
        """Test fetching real EC2 commands (requires network)."""
        result = get_aws_service_commands(service="ec2", use_cache=False)

        assert "describe-instances" in result or "DescribeInstances" in result

    @pytest.mark.integration
    def test_real_lambda_commands(self):
        """Test fetching real Lambda commands (requires network)."""
        result = get_aws_service_commands(service="lambda", use_cache=False)

        assert "invoke" in result or "create-function" in result
