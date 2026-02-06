"""
Tests for the generate_image tool.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands_tools import nova_canvas


@pytest.fixture
def agent():
    """Create an agent with the generate_image tool loaded."""
    return Agent(tools=[nova_canvas])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for testing."""
    with patch("boto3.client") as mock_client:
        # Set up mock response
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"images": [base64.b64encode(b"mock_image_data").decode("utf-8")]}
        ).encode("utf-8")

        mock_client_instance = MagicMock()
        mock_client_instance.invoke_model.return_value = {"body": mock_body}
        mock_client.return_value = mock_client_instance

        yield mock_client


@pytest.fixture
def mock_os_path_exists():
    """Mock os.path.exists for testing."""
    with patch("os.path.exists") as mock_exists:
        # First return False for output directory check, then True for file check to test filename incrementing
        mock_exists.side_effect = [False, True, True, False]
        yield mock_exists


@pytest.fixture
def mock_os_makedirs():
    """Mock os.makedirs for testing."""
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mock_file_open():
    """Mock file open for testing."""
    mock_file = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file

    with patch("builtins.open", return_value=mock_context) as mock_open:
        yield mock_open, mock_file


def test_generate_image_direct(mock_boto3_client, mock_os_path_exists, mock_os_makedirs, mock_file_open):
    """Test direct invocation of the generate_image tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "task_type": "TEXT_IMAGE",
            "text": "A cute robot",
            "seed": 123,
            "negative_text": "blurry, low resolution, pixelated, grainy, unrealistic",
            "style": "DESIGN_SKETCH",
        },
    }

    # Call the generate_image function directly
    result = nova_canvas.nova_canvas(tool=tool_use)

    # Verify the function was called with correct parameters
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.assert_called_once()

    # Check the parameters passed to invoke_model
    args, kwargs = mock_client_instance.invoke_model.call_args
    request_body = json.loads(kwargs["body"])

    assert request_body["textToImageParams"]["text"] == "A cute robot"
    assert request_body["textToImageParams"]["style"] == "DESIGN_SKETCH"
    assert request_body["textToImageParams"]["negativeText"] == "blurry, low resolution, pixelated, grainy, unrealistic"
    assert request_body["imageGenerationConfig"]["seed"] == 123

    # Verify directory creation
    mock_os_makedirs.assert_called_once()

    # Verify file operations
    mock_open, mock_file = mock_file_open
    mock_file.write.assert_called_once()

    # Check the result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "TEXT_IMAGE task completed successfully." in result["content"][0]["text"]
    assert result["content"][1]["image"]["format"] == "png"
    assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)


def test_generate_image_default_params(mock_boto3_client, mock_os_path_exists, mock_os_makedirs, mock_file_open):
    """Test generate_image with default parameters."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"prompt": "A cute robot"}}

    with patch("random.randint", return_value=42):
        result = nova_canvas.nova_canvas(tool=tool_use)

    # Check the default parameters were used
    mock_client_instance = mock_boto3_client.return_value
    args, kwargs = mock_client_instance.invoke_model.call_args
    request_body = json.loads(kwargs["body"])

    assert request_body["imageGenerationConfig"]["seed"] == 42  # From our mocked random.randint
    assert request_body["imageGenerationConfig"]["quality"] == "standard"
    assert result["status"] == "success"


def test_generate_image_error_handling(mock_boto3_client):
    """Test error handling in generate_image."""
    # Setup boto3 client to raise an exception
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.side_effect = Exception("API error")

    tool_use = {"toolUseId": "test-tool-use-id", "input": {"prompt": "A cute robot"}}

    result = nova_canvas.nova_canvas(tool=tool_use)

    # Verify error handling
    assert result["status"] == "error"
    assert "Error generating image: API error" in result["content"][0]["text"]


def test_virtual_try_on(mock_boto3_client, mock_os_path_exists, mock_os_makedirs, mock_file_open):
    """Test virtual try-on functionality."""
    # Mock file reading for images
    with patch("builtins.open", mock_file_open[0]):
        with patch("strands_tools.nova_canvas.encode_image_file") as mock_encode:
            mock_encode.side_effect = ["source_image_b64", "reference_image_b64"]

            tool_use = {
                "toolUseId": "test-tool-use-id",
                "input": {
                    "task_type": "VIRTUAL_TRY_ON",
                    "image_path": "person.jpg",
                    "reference_image_path": "shirt.jpg",
                    "mask_type": "GARMENT",
                    "garment_class": "SHORT_SLEEVE_SHIRT",
                    "preserve_face": "ON",
                },
            }

            result = nova_canvas.nova_canvas(tool=tool_use)

            # Verify the function was called with correct parameters
            mock_client_instance = mock_boto3_client.return_value
            args, kwargs = mock_client_instance.invoke_model.call_args
            request_body = json.loads(kwargs["body"])

            assert request_body["taskType"] == "VIRTUAL_TRY_ON"
            assert request_body["virtualTryOnParams"]["sourceImage"] == "source_image_b64"
            assert request_body["virtualTryOnParams"]["referenceImage"] == "reference_image_b64"
            assert request_body["virtualTryOnParams"]["maskType"] == "GARMENT"
            assert request_body["virtualTryOnParams"]["garmentBasedMask"]["garmentClass"] == "SHORT_SLEEVE_SHIRT"
            assert request_body["virtualTryOnParams"]["maskExclusions"]["preserveFace"] == "ON"

            assert result["status"] == "success"
            assert "VIRTUAL_TRY_ON task completed successfully" in result["content"][0]["text"]


def test_background_removal(mock_boto3_client, mock_os_path_exists, mock_os_makedirs, mock_file_open):
    """Test background removal functionality."""
    # Mock file reading for image
    with patch("builtins.open", mock_file_open[0]):
        with patch("strands_tools.nova_canvas.encode_image_file") as mock_encode:
            mock_encode.return_value = "image_b64_data"

            tool_use = {
                "toolUseId": "test-tool-use-id",
                "input": {"task_type": "BACKGROUND_REMOVAL", "image_path": "photo.jpg"},
            }

            result = nova_canvas.nova_canvas(tool=tool_use)

            # Verify the function was called with correct parameters
            mock_client_instance = mock_boto3_client.return_value
            args, kwargs = mock_client_instance.invoke_model.call_args
            request_body = json.loads(kwargs["body"])

            assert request_body["taskType"] == "BACKGROUND_REMOVAL"
            assert request_body["backgroundRemovalParams"]["image"] == "image_b64_data"

            assert result["status"] == "success"
            assert "BACKGROUND_REMOVAL task completed successfully" in result["content"][0]["text"]


def test_filename_creation():
    """Test the filename creation logic using regex patterns similar to create_filename."""

    # Since create_filename is defined inside the function, we'll replicate its functionality
    def create_filename_test(prompt: str) -> str:
        import re

        words = re.findall(r"\w+", prompt.lower())[:5]
        filename = "_".join(words)
        filename = re.sub(r"[^\w\-_\.]", "_", filename)
        return filename[:100]

    # Test normal prompt
    filename = create_filename_test("A cute robot dancing in the rain")
    assert filename == "a_cute_robot_dancing_in"

    # Test prompt with special characters
    filename = create_filename_test("A cute robot! With @#$% special chars")
    assert filename == "a_cute_robot_with_special"

    # Test long prompt
    long_prompt = "This is a very long prompt " + "word " * 50
    filename = create_filename_test(long_prompt)
    assert len(filename) <= 100


def test_generate_image_via_agent(agent, mock_boto3_client, mock_os_path_exists, mock_os_makedirs, mock_file_open):
    """Test image generation (default tool) via the agent interface."""
    # This simulates how the tool would be used through the Agent interface
    result = agent.tool.nova_canvas(prompt="Test via agent")

    result_text = extract_result_text(result)
    assert "TEXT_IMAGE task completed successfully." in result_text
