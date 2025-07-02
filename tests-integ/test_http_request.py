"""Integration tests for HTTP request functionality using Strands Agent.

This module contains integration tests that verify the HTTP request tool's ability
to handle various HTTP methods (GET, POST, PUT) and operations like downloading
files and interacting with AWS S3 services.
"""

import os

import pytest
from strands import Agent
from strands_tools import http_request


@pytest.fixture
def agent():
    """Create an Agent instance configured with HTTP request tool."""
    return Agent(tools=[http_request])


def test_get_http_request(agent):
    """Test GET request functionality by fetching GitHub repository information.

    Verifies that the agent can successfully make a GET request to the GitHub API
    and retrieve repository star count information.

    """
    response = agent(
        "Send a GET request to https://api.github.com/repos/strands-agents/sdk-python and show me the number of stars"
    )
    print(str(response))
    assert "stars" in str(response).lower()


def test_post_http_request(agent):
    """Test POST request functionality with JSON payload.

    Verifies that the agent can successfully make a POST request with a JSON body
    to httpbin.org and receive the expected response containing the sent data.

    Note:
        Test may be skipped if the external endpoint is unavailable or returns
        a cancelled response due to service instability.
    """
    response = agent('Send a POST request to https://httpbin.org/post with JSON body {"foo": "bar"}')
    if "cancelled" in str(response).lower():
        pytest.skip("This test is flaky due to their endpoint. Skipping for now.")
    assert "foo" in str(response).lower(), str(response)


def test_update_http_request(agent):
    """Test PUT request functionality with JSON payload.

    Verifies that the agent can successfully make a PUT request with a JSON body
    to httpbin.org for update operations and receive the expected response.

    Note:
        Test may be skipped if the external endpoint is unavailable or returns
        a cancelled response due to service instability.
    """
    response = agent('Send a PUT request to https://httpbin.org/put with JSON body {"update": true}')
    if "cancelled" in str(response).lower():
        pytest.skip("This test is flaky due to their endpoint. Skipping for now.")
    assert "update" in str(response).lower(), str(response)


def test_download_http_request(agent):
    """Test file download functionality via HTTP request.

    Verifies that the agent can successfully download a file (PNG image) from
    a remote URL and provide confirmation of the download operation.

    """
    response = agent("Download the PNG logo from https://www.python.org/static/community_logos/python-logo.png")
    assert "successfully" in str(response).lower() or "image" in str(response).lower(), str(response)


@pytest.mark.skipif("AWS_ACCESS_KEY_ID" not in os.environ, reason="ACCESS_KEY_ID environment variable missing")
def test_list_s3_bucket_http_request(agent):
    """Test AWS S3 bucket listing functionality via HTTP requests.

    Verifies that the agent can successfully interact with AWS S3 services
    to list buckets in a specified region using HTTP requests.

    Note:
        This test requires AWS_ACCESS_KEY_ID environment variable to be set.
        The test will be skipped if AWS credentials are not available.
        Uses AWS_REGION environment variable or defaults to 'us-east-1'.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    response = agent(f"List all S3 buckets in region {region}.")
    assert "s3 buckets" in str(response).lower(), str(response)
