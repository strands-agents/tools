"""Tests for the nova_reels tool."""

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
from botocore.config import Config as BotocoreConfig
from strands import Agent

from strands_tools import nova_reels


@pytest.fixture
def agent():
    """Create an agent with the nova_reels tool loaded."""
    return Agent(tools=[nova_reels])


@pytest.fixture
def mock_bedrock_client():
    """Create a mock boto3 client for bedrock-runtime."""
    mock_client = MagicMock()

    # Mock responses for different API calls
    mock_client.start_async_invoke.return_value = {
        "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-inference/test-job-id"
    }

    mock_client.get_async_invoke.return_value = {
        "status": "Completed",
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://test-bucket/output"}},
        "submitTime": "2023-01-01T00:00:00Z",
    }

    mock_client.list_async_invokes.return_value = {
        "asyncInvokeSummaries": [
            {"invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-inference/job1", "status": "Completed"},
            {"invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-inference/job2", "status": "InProgress"},
        ]
    }

    return mock_client


@pytest.fixture
def sample_image_bytes():
    """Return sample image bytes for testing."""
    return b"sample_image_data"


def test_nova_reels_create_text_to_video(mock_bedrock_client):
    """Test creating a text-to-video job."""
    with patch("boto3.client", return_value=mock_bedrock_client) as mock_boto3:
        result = nova_reels.nova_reels(
            action="create",
            text="A cinematic shot of a giraffe walking through a savanna at sunset",
            s3_bucket="test-bucket",
        )

        # Verify result structure
        assert result["status"] == "success"
        assert "Video generation job started successfully" in result["content"][0]["text"]
        assert "Task ARN:" in result["content"][1]["text"]

        # Verify boto3 client was called with correct parameters including user agent
        mock_boto3.assert_called_once()
        args, kwargs = mock_boto3.call_args
        assert args[0] == "bedrock-runtime"

        assert "config" in kwargs
        config = kwargs["config"]
        assert isinstance(config, BotocoreConfig)
        assert config.user_agent_extra == "strands-agents-nova-reels"

        # Verify bedrock client was called with correct parameters
        mock_bedrock_client.start_async_invoke.assert_called_once()
        args = mock_bedrock_client.start_async_invoke.call_args[1]
        assert args["modelId"] == "amazon.nova-reel-v1:1"
        assert "TEXT_VIDEO" in str(args["modelInput"])
        assert "test-bucket" in str(args["outputDataConfig"])


def test_nova_reels_create_image_to_video(mock_bedrock_client, sample_image_bytes):
    """Test creating an image-to-video job."""
    m = mock_open(read_data=sample_image_bytes)

    with patch("boto3.client", return_value=mock_bedrock_client), patch("builtins.open", m):
        result = nova_reels.nova_reels(
            action="create",
            text="Transform this forest into autumn",
            image_path="/path/to/image.jpg",
            s3_bucket="test-bucket",
            seed=42,
            fps=30,
            dimension="1920x1080",
        )

        # Verify result
        assert result["status"] == "success"

        # Verify boto3 client was called with correct parameters
        mock_bedrock_client.start_async_invoke.assert_called_once()
        args = mock_bedrock_client.start_async_invoke.call_args[1]

        # Verify video configuration
        model_input = args["modelInput"]
        assert "images" in str(model_input)
        assert "fps" in str(model_input)
        assert "30" in str(model_input)  # Check fps
        assert "1920x1080" in str(model_input)  # Check dimension
        assert "42" in str(model_input)  # Check seed


def test_nova_reels_status(mock_bedrock_client):
    """Test checking the status of a video generation job."""
    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(
            action="status", invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-inference/test-job-id"
        )

        # Verify result
        assert result["status"] == "success"
        assert "Video generation completed!" in result["content"][0]["text"]
        assert "s3://test-bucket/output/output.mp4" in result["content"][1]["text"]

        # Verify boto3 client was called correctly
        mock_bedrock_client.get_async_invoke.assert_called_once_with(
            invocationArn="arn:aws:bedrock:us-east-1:123456789012:async-inference/test-job-id"
        )


def test_nova_reels_status_in_progress(mock_bedrock_client):
    """Test checking a job that's still in progress."""
    mock_bedrock_client.get_async_invoke.return_value = {"status": "InProgress", "submitTime": "2023-01-01T00:00:00Z"}

    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(
            action="status", invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-inference/in-progress-job"
        )

        # Verify result
        assert result["status"] == "success"
        assert "Job in progress" in result["content"][0]["text"]
        assert "Started at:" in result["content"][1]["text"]


def test_nova_reels_status_failed(mock_bedrock_client):
    """Test checking a failed job."""
    mock_bedrock_client.get_async_invoke.return_value = {"status": "Failed", "failureMessage": "Resource not found"}

    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(
            action="status", invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-inference/failed-job"
        )

        # Verify result
        assert result["status"] == "success"
        assert "Job failed" in result["content"][0]["text"]
        assert "Resource not found" in result["content"][1]["text"]


def test_nova_reels_list(mock_bedrock_client):
    """Test listing video generation jobs."""
    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(action="list", max_results=5)

        # Verify result
        assert result["status"] == "success"
        assert "Found 2 jobs" in result["content"][0]["text"]

        # Verify boto3 client was called correctly
        mock_bedrock_client.list_async_invokes.assert_called_once_with(maxResults=5)


def test_nova_reels_list_with_status_filter(mock_bedrock_client):
    """Test listing jobs with a status filter."""
    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(action="list", max_results=5, status_filter="Completed")

        # Verify result
        assert result["status"] == "success"

        # Verify boto3 client was called correctly
        mock_bedrock_client.list_async_invokes.assert_called_once_with(maxResults=5, statusEquals="Completed")


def test_nova_reels_custom_region(mock_bedrock_client):
    """Test using a custom region."""
    with patch("boto3.client", return_value=mock_bedrock_client) as mock_boto3:
        nova_reels.nova_reels(action="list", max_results=5, region="us-west-2")

        # Verify boto3 client was created with the right region and user agent
        mock_boto3.assert_called_once()
        args, kwargs = mock_boto3.call_args
        assert args[0] == "bedrock-runtime"
        assert kwargs["region_name"] == "us-west-2"
        assert "config" in kwargs
        config = kwargs["config"]
        assert isinstance(config, BotocoreConfig)
        assert config.user_agent_extra == "strands-agents-nova-reels"


def test_nova_reels_region_from_env(mock_bedrock_client):
    """Test using a region from environment variable."""
    with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
        with patch("boto3.client", return_value=mock_bedrock_client) as mock_boto3:
            nova_reels.nova_reels(action="list", max_results=5)

            # Verify boto3 client was created with the right region and user agent
            mock_boto3.assert_called_once()
            args, kwargs = mock_boto3.call_args
            assert args[0] == "bedrock-runtime"
            assert kwargs["region_name"] == "eu-west-1"
            assert "config" in kwargs
            config = kwargs["config"]
            assert isinstance(config, BotocoreConfig)
            assert config.user_agent_extra == "strands-agents-nova-reels"


def test_nova_reels_missing_required_params():
    """Test error handling for missing required parameters."""
    # Test missing text for create action
    result = nova_reels.nova_reels(action="create", s3_bucket="test-bucket")
    assert result["status"] == "error"
    assert "Text prompt is required" in result["content"][0]["text"]

    # Test missing s3_bucket for create action
    result = nova_reels.nova_reels(action="create", text="Test prompt")
    assert result["status"] == "error"
    assert "S3 bucket is required" in result["content"][0]["text"]

    # Test missing invocation_arn for status action
    result = nova_reels.nova_reels(action="status")
    assert result["status"] == "error"
    assert "invocation_arn is required" in result["content"][0]["text"]


def test_nova_reels_invalid_action():
    """Test error handling for invalid action."""
    result = nova_reels.nova_reels(action="invalid_action")
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_nova_reels_invalid_parameters():
    """Test error handling for invalid parameters."""
    # Test invalid dimension format
    result = nova_reels.nova_reels(action="create", text="Test prompt", s3_bucket="test-bucket", dimension="invalid")
    assert result["status"] == "error"
    assert "dimension must be in format" in result["content"][0]["text"]


def test_nova_reels_image_error_handling(mock_bedrock_client):
    """Test error handling for image processing."""
    with (
        patch("boto3.client", return_value=mock_bedrock_client),
        patch("builtins.open", side_effect=FileNotFoundError("File not found")),
    ):
        result = nova_reels.nova_reels(
            action="create",
            text="Transform this forest",
            image_path="/path/to/nonexistent.jpg",
            s3_bucket="test-bucket",
        )

        assert result["status"] == "error"
        assert "Failed to process input image" in result["content"][0]["text"]


def test_nova_reels_status_unknown(mock_bedrock_client):
    """Test status with an unknown job status."""
    # Set an unknown status to cover the uncovered branch
    mock_bedrock_client.get_async_invoke.return_value = {
        "status": "Unknown",  # This should trigger the default branch
        "submitTime": "2023-01-01T00:00:00Z",
    }

    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(
            action="status", invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-inference/unknown-status-job"
        )

        # Verify result - should still return a successful response
        assert result["status"] == "success"
        # But content should be empty (since there's no specific handling for unknown status)
        assert len(result["content"]) == 0


def test_nova_reels_boto3_error_handling(mock_bedrock_client):
    """Test error handling for boto3 client errors."""
    mock_bedrock_client.start_async_invoke.side_effect = Exception("AWS service error")

    with patch("boto3.client", return_value=mock_bedrock_client):
        result = nova_reels.nova_reels(action="create", text="Test prompt", s3_bucket="test-bucket")

        assert result["status"] == "error"
        assert "Error: AWS service error" in result["content"][0]["text"]


def test_nova_reels_via_agent(agent, mock_bedrock_client):
    """Test nova_reels via the agent interface."""
    with patch("boto3.client", return_value=mock_bedrock_client):
        result = agent.tool.nova_reels(
            action="create", text="A cinematic shot of mountains", s3_bucket="agent-test-bucket"
        )

        # Verify result
        assert "status" in result
        assert result["status"] == "success"
        assert "Video generation job started successfully" in result["content"][0]["text"]
