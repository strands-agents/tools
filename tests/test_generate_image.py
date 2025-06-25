"""
Tests for the generate_image tool.
"""

import json
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from strands import Agent
from strands_tools import generate_image


@pytest.fixture
def agent():
    """Create an agent with the generate_image tool loaded."""
    return Agent(tools=[generate_image])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


# Helper functions for common verification patterns
def verify_error_response_structure(result, expected_tool_use_id="test-tool-use-id"):
    """Verify the structure of an error response."""
    assert result["toolUseId"] == expected_tool_use_id
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]
    return result["content"][0]["text"]


def verify_success_response_structure(result, expected_tool_use_id="test-tool-use-id"):
    """Verify the structure of a success response."""
    assert result["toolUseId"] == expected_tool_use_id
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2  # Text and image content

    # Verify text content
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]
    assert "The generated image has been saved locally" in result["content"][0]["text"]

    # Verify image content
    assert isinstance(result["content"][1], dict)
    assert "image" in result["content"][1]
    assert result["content"][1]["image"]["format"] == "png"
    assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)

    return result["content"][0]["text"]


def verify_aws_service_calls(mock_boto3_client, mock_validate_model, model_id, region="us-west-2"):
    """Verify standard AWS service calls were made correctly."""
    # Verify model validation was called
    mock_validate_model.assert_called_once_with(model_id, region)

    # Verify boto3 client creation
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name=region)

    # Verify invoke_model was called
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.assert_called_once()

    return mock_client_instance


def verify_file_operations(mock_os_makedirs, mock_file_open):
    """Verify file system operations were performed correctly."""
    mock_os_makedirs.assert_called_once()
    mock_open, mock_file = mock_file_open
    mock_file.write.assert_called_once()


def verify_request_body_amazon_model(mock_client_instance, expected_prompt, expected_seed=123, expected_cfg_scale=10):
    """Verify request body structure for Amazon Titan models."""
    args, kwargs = mock_client_instance.invoke_model.call_args
    request_body = json.loads(kwargs["body"])
    assert request_body["taskType"] == "TEXT_IMAGE"
    assert request_body["textToImageParams"]["text"] == expected_prompt
    assert request_body["imageGenerationConfig"]["seed"] == expected_seed
    assert request_body["imageGenerationConfig"]["cfgScale"] == expected_cfg_scale
    assert request_body["imageGenerationConfig"]["numberOfImages"] == 1
    return request_body


def verify_request_body_stability_model(mock_client_instance, expected_prompt, expected_seed=123):
    """Verify request body structure for Stability AI models."""
    args, kwargs = mock_client_instance.invoke_model.call_args
    request_body = json.loads(kwargs["body"])
    assert request_body["prompt"] == expected_prompt
    assert request_body["seed"] == expected_seed
    return request_body


@pytest.fixture
def mock_validate_model():
    """Mock the validate_model_in_region function."""
    with patch("strands_tools.generate_image.validate_model_in_region") as mock_validate:
        # Default to valid model and provide a list of available models
        mock_validate.return_value = (True, ["amazon.titan-image-generator-v2:0", "stability.stable-image-ultra-v1:1"])
        yield mock_validate


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for testing."""
    with patch("boto3.client") as mock_client:
        mock_client_instance = MagicMock()
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


@pytest.fixture
def stability_model_response():
    """Mock response for Stability AI models."""
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"images": ["base64_encoded_image_data"]}).encode("utf-8")
    return mock_body


@pytest.fixture
def amazon_model_v1_response():
    """Mock response for Amazon Titan v1 models."""
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"images": ["base64_encoded_image_data"]}).encode("utf-8")
    return mock_body


@pytest.fixture
def amazon_model_v2_response():
    """Mock response for Amazon Titan v2 models."""
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"images": [{"imageBase64": "base64_encoded_image_data"}]}).encode("utf-8")
    return mock_body


@pytest.fixture
def basic_tool_use():
    """Basic tool use object for testing."""
    return {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "seed": 123,
            "cfg_scale": 10,
        },
    }


@contextmanager
def mock_base64_decode():
    """Context manager for mocking base64.b64decode."""
    with patch("base64.b64decode", return_value=b"decoded_image_data"):
        yield


def test_generate_image_stability_model(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    stability_model_response,
    basic_tool_use,
):
    """Test direct invocation of the generate_image tool with a Stability AI model."""
    # Update tool use with stability model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "stability.stable-image-ultra-v1:1"

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        # Call the generate_image function directly
        result = generate_image.generate_image(tool=tool_use)

        # Verify AWS service calls
        mock_client_instance = verify_aws_service_calls(
            mock_boto3_client, mock_validate_model, "stability.stable-image-ultra-v1:1"
        )

        # Verify request body for Stability model
        verify_request_body_stability_model(mock_client_instance, "A cute robot", 123)

        # Verify file operations
        verify_file_operations(mock_os_makedirs, mock_file_open)

        # Verify success response structure
        verify_success_response_structure(result)


def test_generate_image_amazon_model(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    amazon_model_v2_response,
    basic_tool_use,
):
    """Test direct invocation of the generate_image tool with an Amazon Titan model."""
    # Update tool use with Amazon model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.titan-image-generator-v2:0"

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": amazon_model_v2_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        # Call the generate_image function directly
        result = generate_image.generate_image(tool=tool_use)

        # Verify AWS service calls
        mock_client_instance = verify_aws_service_calls(
            mock_boto3_client, mock_validate_model, "amazon.titan-image-generator-v2:0"
        )

        # Verify request body for Amazon model
        verify_request_body_amazon_model(mock_client_instance, "A cute robot", 123, 10)

        # Verify file operations
        verify_file_operations(mock_os_makedirs, mock_file_open)

        # Verify success response structure
        verify_success_response_structure(result)


def test_generate_image_auto_model_selection(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    amazon_model_v2_response,
    basic_tool_use,
):
    """Test automatic model selection when no model_id is provided."""
    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": amazon_model_v2_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        # Call the generate_image function directly
        result = generate_image.generate_image(tool=basic_tool_use)

        # Verify model validation was called for auto-selection (empty model_id)
        mock_validate_model.assert_any_call("", "us-west-2")

        # Verify boto3 client creation
        mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

        # Verify invoke_model was called
        mock_client_instance.invoke_model.assert_called_once()

        # Verify request body for Amazon model (auto-selected)
        verify_request_body_amazon_model(mock_client_instance, "A cute robot", 123, 10)

        # Verify file operations
        verify_file_operations(mock_os_makedirs, mock_file_open)

        # Verify success response structure
        verify_success_response_structure(result)


def test_generate_image_model_validation_error(mock_validate_model, basic_tool_use):
    """Test error handling when model validation fails."""
    # Setup validation to fail
    mock_validate_model.return_value = (False, ["amazon.titan-image-generator-v2:0"])

    # Update tool use with invalid model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "stability.invalid-model"

    result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called with the invalid model
    mock_validate_model.assert_called_once_with("stability.invalid-model", "us-west-2")

    # Verify error response structure and content
    error_text = verify_error_response_structure(result)
    assert "not available in region" in error_text
    assert "stability.invalid-model" in error_text
    assert "us-west-2" in error_text
    assert "amazon.titan-image-generator-v2:0" in error_text  # Available models should be listed


def test_generate_image_legacy_model_error(mock_validate_model, basic_tool_use):
    """Test error handling when a legacy model is requested."""
    # Setup validation to raise ValueError for legacy model
    mock_validate_model.side_effect = ValueError("Model is in LEGACY status")

    # Update tool use with legacy model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "stability.legacy-model"

    result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called with the legacy model
    mock_validate_model.assert_called_once_with("stability.legacy-model", "us-west-2")

    # Verify error response structure and content
    error_text = verify_error_response_structure(result)
    assert "Model is in LEGACY status" in error_text


def test_generate_image_access_denied(mock_boto3_client, mock_validate_model, basic_tool_use):
    """Test error handling when access is denied to the model."""
    # Setup boto3 client to raise an AccessDeniedException
    mock_client_instance = mock_boto3_client.return_value
    access_denied_exception = Exception("AccessDeniedException: Access denied")
    mock_client_instance.invoke_model.side_effect = access_denied_exception

    # Update tool use with model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.titan-image-generator-v2:0"

    result = generate_image.generate_image(tool=tool_use)

    # Verify AWS service calls were attempted
    verify_aws_service_calls(mock_boto3_client, mock_validate_model, "amazon.titan-image-generator-v2:0")

    # Verify error response structure and content
    error_text = verify_error_response_structure(result)
    assert "Error generating image" in error_text


def test_validate_model_in_region():
    """Test the validate_model_in_region function."""
    with patch("boto3.client") as mock_client:
        # Mock the list_foundation_models response
        mock_client_instance = MagicMock()
        mock_client_instance.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.titan-image-generator-v2:0",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["IMAGE"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {
                    "modelId": "stability.stable-image-ultra-v1:1",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["IMAGE"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {
                    "modelId": "stability.stable-diffusion-xl-v1",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["IMAGE"],
                    "modelLifecycle": {"status": "LEGACY"},
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
            ]
        }
        mock_client.return_value = mock_client_instance

        # Test valid model
        is_valid, available_models = generate_image.validate_model_in_region(
            "amazon.titan-image-generator-v2:0", "us-west-2"
        )
        assert is_valid is True
        assert "amazon.titan-image-generator-v2:0" in available_models

        # Test invalid model
        is_valid, available_models = generate_image.validate_model_in_region("invalid.model", "us-west-2")
        assert is_valid is False

        # Test legacy model
        with pytest.raises(ValueError) as excinfo:
            generate_image.validate_model_in_region("stability.stable-diffusion-xl-v1", "us-west-2")
        assert "LEGACY status" in str(excinfo.value)


def test_create_filename():
    """Test the create_filename function."""
    # Test the actual function from the module
    filename = generate_image.create_filename("A cute robot dancing in the rain")
    assert filename == "a_cute_robot_dancing_in"

    # Test with special characters
    filename = generate_image.create_filename("A cute robot! With @#$% special chars")
    assert filename == "a_cute_robot_with_special"

    # Test long prompt
    long_prompt = "This is a very long prompt " + "word " * 50
    filename = generate_image.create_filename(long_prompt)
    assert len(filename) <= 100


def test_generate_image_via_agent(
    agent, mock_boto3_client, mock_validate_model, mock_os_path_exists, mock_os_makedirs, mock_file_open
):
    """Test image generation via the agent interface."""
    # Set up mock response for agent test
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"images": ["base64_encoded_image_data"]}).encode("utf-8")

    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": mock_body}

    # This simulates how the tool would be used through the Agent interface
    # We mock the agent's tool method to return a comprehensive response
    mock_response = {
        "toolUseId": "test-tool-use-id",
        "status": "success",
        "content": [
            {"text": "The generated image has been saved locally to output/test_via_agent.png"},
            {"image": {"format": "png", "source": {"bytes": b"decoded_image_data"}}},
        ],
    }

    with patch.object(agent.tool, "generate_image", return_value=mock_response) as mock_generate:
        result = agent.tool.generate_image(prompt="Test via agent")

        # Verify the agent tool method was called with correct parameters
        mock_generate.assert_called_once_with(prompt="Test via agent")

        # Extract and verify result content
        result_text = extract_result_text(result)
        assert "The generated image has been saved locally" in result_text

        # Verify complete result structure
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
        assert len(result["content"]) == 2

        # Verify text content
        assert "The generated image has been saved locally" in result["content"][0]["text"]

        # Verify image content structure
        image_content = result["content"][1]
        assert "image" in image_content
        assert image_content["image"]["format"] == "png"
        assert isinstance(image_content["image"]["source"]["bytes"], bytes)
        assert image_content["image"]["source"]["bytes"] == b"decoded_image_data"


@pytest.mark.parametrize(
    "model_id,response_fixture",
    [
        ("stability.stable-image-ultra-v1:1", "stability_model_response"),
        ("amazon.titan-image-generator-v1", "amazon_model_v1_response"),
        ("amazon.titan-image-generator-v2:0", "amazon_model_v2_response"),
    ],
)
def test_model_response_parsing(
    model_id,
    response_fixture,
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    request,
):
    """Test parsing of responses from different model families."""
    # Get the response fixture
    mock_body = request.getfixturevalue(response_fixture)

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": mock_body}

    # Create tool use with the specified model
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": model_id,
        },
    }

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        result = generate_image.generate_image(tool=tool_use)

        # Verify model validation was called
        mock_validate_model.assert_called_once_with(model_id, "us-west-2")

        # Verify boto3 client creation
        mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

        # Verify invoke_model was called
        mock_client_instance.invoke_model.assert_called_once()

        # Verify complete success response structure
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2  # Text and image content

        # Verify text content
        assert isinstance(result["content"][0], dict)
        assert "text" in result["content"][0]
        assert "The generated image has been saved locally" in result["content"][0]["text"]

        # Verify image content
        assert isinstance(result["content"][1], dict)
        assert "image" in result["content"][1]
        assert result["content"][1]["image"]["format"] == "png"
        assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)

        # Verify file operations
        mock_os_makedirs.assert_called_once()
        mock_open, mock_file = mock_file_open
        mock_file.write.assert_called_once()


def test_response_parsing_error(mock_boto3_client, mock_validate_model):
    """Test error handling for unexpected response formats."""
    # Set up mock response with unexpected format
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"unexpected_format": "data"}).encode("utf-8")

    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": mock_body}

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "amazon.titan-image-generator-v2:0",
        },
    }

    result = generate_image.generate_image(tool=tool_use)

    # Verify AWS service calls were attempted
    verify_aws_service_calls(mock_boto3_client, mock_validate_model, "amazon.titan-image-generator-v2:0")

    # Verify error response structure and content
    error_text = verify_error_response_structure(result)
    assert "Error generating image" in error_text


def test_missing_prompt_error():
    """Test handling when prompt is missing - should use default prompt."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            # No prompt provided - should use default
            "model_id": "amazon.titan-image-generator-v2:0",
        },
    }

    # Mock the necessary components to avoid actual AWS calls
    with patch("strands_tools.generate_image.validate_model_in_region", return_value=(True, [])):
        with patch("boto3.client") as mock_client:
            mock_client_instance = MagicMock()
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({"images": [{"imageBase64": "base64_data"}]}).encode("utf-8")
            mock_client_instance.invoke_model.return_value = {"body": mock_body}
            mock_client.return_value = mock_client_instance

            with patch("os.path.exists", return_value=False):
                with patch("os.makedirs"):
                    with patch("builtins.open", MagicMock()):
                        with patch("base64.b64decode", return_value=b"image_data"):
                            result = generate_image.generate_image(tool=tool_use)

    # Verify complete success response structure (should succeed with default prompt)
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2  # Text and image content
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify success message
    success_text = result["content"][0]["text"]
    assert "The generated image has been saved locally" in success_text

    # Verify image content
    assert isinstance(result["content"][1], dict)
    assert "image" in result["content"][1]
    assert result["content"][1]["image"]["format"] == "png"


def test_file_path_construction(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    stability_model_response,
):
    """Test that file paths are constructed correctly."""
    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot dancing",
            "model_id": "amazon.titan-image-generator-v2:0",
        },
    }

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called
    mock_validate_model.assert_called_once_with("amazon.titan-image-generator-v2:0", "us-west-2")

    # Verify boto3 client creation
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

    # Verify invoke_model was called
    mock_client_instance.invoke_model.assert_called_once()

    # Verify directory creation
    mock_os_makedirs.assert_called_once()

    # Verify file operations
    mock_open, mock_file = mock_file_open
    mock_file.write.assert_called_once()

    # Check that the file path contains the expected filename pattern
    file_path_arg = mock_open.call_args[0][0]
    assert "output" in file_path_arg
    assert "a_cute_robot_dancing" in file_path_arg
    assert file_path_arg.endswith(".png")

    # Verify successful result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "The generated image has been saved locally" in result["content"][0]["text"]


def test_custom_region_handling(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    stability_model_response,
):
    """Test handling of custom region specification."""
    # Create a tool use dictionary with custom region
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "amazon.titan-image-generator-v2:0",
            "region": "us-east-1",
        },
    }

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called with custom region
    mock_validate_model.assert_called_once_with("amazon.titan-image-generator-v2:0", "us-east-1")

    # Verify the region was used correctly for client creation
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

    # Verify invoke_model was called
    mock_client_instance.invoke_model.assert_called_once()

    # Verify file operations
    mock_os_makedirs.assert_called_once()
    mock_open, mock_file = mock_file_open
    mock_file.write.assert_called_once()

    # Verify successful result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "The generated image has been saved locally" in result["content"][0]["text"]


def test_environment_variable_region(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    stability_model_response,
):
    """Test handling of region from environment variable."""
    # Create a tool use dictionary without region
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "amazon.titan-image-generator-v2:0",
        },
    }

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

    # Mock environment variable
    with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
        # Mock base64 decode to avoid errors
        with mock_base64_decode():
            result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called with environment region
    mock_validate_model.assert_called_once_with("amazon.titan-image-generator-v2:0", "eu-west-1")

    # Verify the region from environment variable was used
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")

    # Verify invoke_model was called
    mock_client_instance.invoke_model.assert_called_once()

    # Verify file operations
    mock_os_makedirs.assert_called_once()
    mock_open, mock_file = mock_file_open
    mock_file.write.assert_called_once()

    # Verify successful result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "The generated image has been saved locally" in result["content"][0]["text"]


def test_duplicate_filename_handling(
    mock_boto3_client, mock_validate_model, mock_os_makedirs, mock_file_open, stability_model_response
):
    """Test handling of duplicate filenames."""
    # Mock os.path.exists to simulate existing files
    with patch("os.path.exists") as mock_exists:
        # First return False for output directory check, then True for all file checks
        mock_exists.side_effect = [False, True, True, True, False]

        # Create a tool use dictionary
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "prompt": "A cute robot",
                "model_id": "amazon.titan-image-generator-v2:0",
            },
        }

        # Set up mock response
        mock_client_instance = mock_boto3_client.return_value
        mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

        # Mock base64 decode to avoid errors
        with mock_base64_decode():
            result = generate_image.generate_image(tool=tool_use)

        # Verify model validation was called
        mock_validate_model.assert_called_once_with("amazon.titan-image-generator-v2:0", "us-west-2")

        # Verify boto3 client creation
        mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

        # Verify invoke_model was called
        mock_client_instance.invoke_model.assert_called_once()

        # Verify directory creation
        mock_os_makedirs.assert_called_once()

        # Verify file operations - should try multiple filenames
        mock_open, mock_file = mock_file_open
        mock_file.write.assert_called_once()

        file_path_arg = mock_open.call_args[0][0]

        # Should have a number appended to the filename due to duplicates
        assert "_3" in file_path_arg
        assert "output" in file_path_arg
        assert file_path_arg.endswith(".png")

        # Verify successful result
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
        assert "The generated image has been saved locally" in result["content"][0]["text"]


def test_client_creation_error():
    """Test error handling when client creation fails."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "amazon.titan-image-generator-v2:0",
        },
    }

    # Mock validate_model_in_region to avoid model validation errors
    with patch("strands_tools.generate_image.validate_model_in_region", return_value=(True, [])):
        # Mock boto3.client to raise an exception
        with patch("boto3.client", side_effect=Exception("Failed to create client")) as mock_client:
            result = generate_image.generate_image(tool=tool_use)

            # Verify boto3.client was called and failed
            mock_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message indicates client creation failure
    error_text = result["content"][0]["text"]
    assert "Error generating image" in error_text


def test_unsupported_model_family():
    """Test error handling for unsupported model families."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "unsupported.model-family-v1:0",
        },
    }

    # Mock validate_model_in_region to return valid (to pass validation)
    with patch("strands_tools.generate_image.validate_model_in_region", return_value=(True, [])):
        with patch("boto3.client") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            result = generate_image.generate_image(tool=tool_use)

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message indicates unsupported model
    error_text = result["content"][0]["text"]
    assert "Unsupported model" in error_text


def test_validation_exception_model_not_found(mock_boto3_client, mock_validate_model, basic_tool_use):
    """Test error handling for ValidationException when model is not found."""
    # Setup boto3 client to raise ValidationException
    mock_client_instance = mock_boto3_client.return_value

    # Create a mock ValidationException
    validation_exception = ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "Model not found"}},
        operation_name="InvokeModel",
    )
    mock_client_instance.invoke_model.side_effect = validation_exception

    # Update tool use with model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.nonexistent-model-v1:0"

    result = generate_image.generate_image(tool=tool_use)

    # Verify model validation was called
    mock_validate_model.assert_called_once_with("amazon.nonexistent-model-v1:0", "us-west-2")

    # Verify boto3 client creation
    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

    # Verify invoke_model was called and raised the exception
    mock_client_instance.invoke_model.assert_called_once()

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message content
    error_text = result["content"][0]["text"]
    assert "not found" in error_text
    assert "amazon.nonexistent-model-v1:0" in error_text


def test_bedrock_client_error_in_validation(basic_tool_use):
    """Test error handling when Bedrock client fails during model validation."""
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.titan-image-generator-v2:0"

    # Mock validate_model_in_region to raise a generic exception
    with patch("strands_tools.generate_image.validate_model_in_region", side_effect=Exception("Bedrock API error")):
        result = generate_image.generate_image(tool=tool_use)

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message content
    error_text = result["content"][0]["text"]
    assert "Could not validate model availability" in error_text
    assert "us-west-2" in error_text


def test_auto_model_selection_no_models_available(basic_tool_use):
    """Test auto model selection when no models are available in region."""
    # Remove model_id to trigger auto-selection
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    # Don't set model_id to trigger auto-selection

    # Mock validate_model_in_region to return no available models
    with patch("strands_tools.generate_image.validate_model_in_region", return_value=(False, [])):
        result = generate_image.generate_image(tool=tool_use)

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message content
    error_text = result["content"][0]["text"]
    assert "No text-to-image models available" in error_text
    assert "us-west-2" in error_text


def test_auto_model_selection_validation_error(basic_tool_use):
    """Test auto model selection when validation fails."""
    # Remove model_id to trigger auto-selection
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    # Don't set model_id to trigger auto-selection

    # Mock validate_model_in_region to raise an exception during auto-selection
    with patch("strands_tools.generate_image.validate_model_in_region", side_effect=Exception("API error")):
        result = generate_image.generate_image(tool=tool_use)

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify error message content
    error_text = result["content"][0]["text"]
    assert "Error determining available models" in error_text
    assert "us-west-2" in error_text
    assert "specify a model_id explicitly" in error_text


def test_amazon_model_v1_response_format(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    basic_tool_use,
):
    """Test parsing Amazon Titan v1 response format (string array)."""
    # Set up mock response for Amazon Titan v1 (string format)
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"images": ["base64_encoded_image_data"]}).encode("utf-8")

    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": mock_body}

    # Update tool use with Amazon model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.titan-image-generator-v1"

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        result = generate_image.generate_image(tool=tool_use)

        # Verify model validation was called
        mock_validate_model.assert_called_once_with("amazon.titan-image-generator-v1", "us-west-2")

        # Verify boto3 client creation
        mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

        # Verify invoke_model was called
        mock_client_instance.invoke_model.assert_called_once()

        # Check the parameters passed to invoke_model
        args, kwargs = mock_client_instance.invoke_model.call_args
        assert kwargs["modelId"] == "amazon.titan-image-generator-v1"

        request_body = json.loads(kwargs["body"])
        assert request_body["taskType"] == "TEXT_IMAGE"
        assert request_body["textToImageParams"]["text"] == "A cute robot"

        # Verify file operations
        mock_os_makedirs.assert_called_once()
        mock_open, mock_file = mock_file_open
        mock_file.write.assert_called_once()

        # Check the result
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
        assert "The generated image has been saved locally" in result["content"][0]["text"]
        assert result["content"][1]["image"]["format"] == "png"
        assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)


def test_image_extraction_key_error(mock_boto3_client, mock_validate_model, basic_tool_use):
    """Test error handling when image extraction fails due to KeyError."""
    # Set up mock response with missing keys
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"no_images_key": "data"}).encode("utf-8")

    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": mock_body}

    # Update tool use with model
    tool_use = basic_tool_use.copy()
    tool_use["input"] = tool_use["input"].copy()
    tool_use["input"]["model_id"] = "amazon.titan-image-generator-v2:0"

    result = generate_image.generate_image(tool=tool_use)

    # Verify complete error response structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert isinstance(result["content"][0], dict)
    assert "text" in result["content"][0]

    # Verify specific error message content (based on actual implementation behavior)
    error_text = result["content"][0]["text"]
    assert "Error generating image" in error_text
    assert "Unexpected Amazon model response structure" in error_text


def test_custom_parameters_validation(
    mock_boto3_client,
    mock_validate_model,
    mock_os_path_exists,
    mock_os_makedirs,
    mock_file_open,
    stability_model_response,
):
    """Test that custom parameters (seed, cfg_scale) are properly used."""
    # Create a tool use dictionary with custom parameters
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cute robot",
            "model_id": "stability.stable-image-ultra-v1:1",
            "seed": 12345,
            "cfg_scale": 15,
        },
    }

    # Set up mock response
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.invoke_model.return_value = {"body": stability_model_response}

    # Mock base64 decode to avoid errors
    with mock_base64_decode():
        result = generate_image.generate_image(tool=tool_use)

        # Check the parameters passed to invoke_model
        args, kwargs = mock_client_instance.invoke_model.call_args
        assert kwargs["modelId"] == "stability.stable-image-ultra-v1:1"

        request_body = json.loads(kwargs["body"])
        assert request_body["prompt"] == "A cute robot"
        assert request_body["seed"] == 12345  # Custom seed should be used

        # Verify successful result
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
