"""
Tests for the generate_image_stability tool.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands_tools import generate_image_stability


@pytest.fixture
def agent():
    """Create an agent with the generate_image_stability tool loaded."""
    return Agent(tools=[generate_image_stability])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@pytest.fixture
def mock_requests():
    """Mock requests for testing Stability API calls."""
    with patch("requests.post") as mock_post:
        # Set up mock response for image return type
        mock_response = MagicMock()
        mock_response.content = b"mock_image_data"
        mock_response.headers = {"finish_reason": "SUCCESS"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        yield mock_post, mock_response


@pytest.fixture
def mock_requests_json():
    """Mock requests for testing Stability API calls with JSON return type."""
    with patch("requests.post") as mock_post:
        # Set up mock response for JSON return type
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "image": base64.b64encode(b"mock_image_data").decode("utf-8"),
            "finish_reason": "SUCCESS",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        yield mock_post, mock_response


def test_generate_image_stability_direct_image_type(mock_requests):
    """Test direct invocation of the generate_image_stability tool with image return type."""
    mock_post, mock_response = mock_requests

    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A futuristic robot",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
            "return_type": "image",
            "seed": 123,
            "cfg_scale": 7.0,
            "aspect_ratio": "16:9",
            "style_preset": "cinematic",
            "output_format": "png",
        },
    }

    # Call the generate_image_stability function directly
    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify the API was called correctly
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args

    # Check URL for SD3 model
    assert args[0] == "https://api.stability.ai/v2beta/stable-image/generate/core"

    # Check headers
    assert kwargs["headers"]["authorization"] == "Bearer sk-test-key"
    assert kwargs["headers"]["accept"] == "image/*"

    # Check data parameters
    data = kwargs["data"]
    assert data["prompt"] == "A futuristic robot"
    assert data["output_format"] == "png"
    assert data["aspect_ratio"] == "16:9"
    assert data["cfg_scale"] == 7.0
    assert data["seed"] == 123
    assert data["style_preset"] == "cinematic"

    # Check files parameter
    assert "none" in kwargs["files"]

    # Check the result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Generated image using stability.stable-image-core-v1:1" in result["content"][0]["text"]
    assert "Finish reason: SUCCESS" in result["content"][0]["text"]
    assert result["content"][1]["image"]["format"] == "png"
    assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)


def test_model_endpoint_routing(mock_requests):
    """Test that different models route to correct API endpoints."""
    mock_post, mock_response = mock_requests

    model_endpoint_tests = [
        ("stability.sd3-5-large-v1:0", "https://api.stability.ai/v2beta/stable-image/generate/sd3"),
        ("stability.stable-image-core-v1:1", "https://api.stability.ai/v2beta/stable-image/generate/core"),
        ("stability.stable-image-ultra-v1:1", "https://api.stability.ai/v2beta/stable-image/generate/ultra"),
    ]

    for model_id, expected_url in model_endpoint_tests:
        # Reset mock for each test
        mock_post.reset_mock()

        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "prompt": "Test prompt",
                "model_id": model_id,
                "stability_api_key": "sk-test-key",
            },
        }

        result = generate_image_stability.generate_image_stability(tool=tool_use)

        # Check that the correct URL was called
        args, kwargs = mock_post.call_args
        assert args[0] == expected_url
        assert result["status"] == "success"


def test_unsupported_model_error():
    """Test error handling for unsupported model_id."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "Test prompt",
            "model_id": "unsupported.model.v1:0",
            "stability_api_key": "sk-test-key",
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify error handling for unsupported model
    assert result["status"] == "error"
    assert "Unsupported model_id: unsupported.model.v1:0" in result["content"][0]["text"]


def test_generate_image_stability_json_type(mock_requests_json):
    """Test generate_image_stability with JSON return type."""
    mock_post, mock_response = mock_requests_json

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A cyberpunk city",
            "model_id": "stability.sd3-5-large-v1:0",
            "stability_api_key": "sk-test-key",
            "return_type": "json",
            "cfg_scale": 8.0,
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify the API was called with JSON accept header
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["accept"] == "application/json"

    # Check that cfg_scale was included in the request
    assert kwargs["data"]["cfg_scale"] == 8.0

    # Check the result
    assert result["status"] == "success"
    assert isinstance(result["content"][1]["image"]["source"]["bytes"], bytes)


def test_generate_image_stability_default_params(mock_requests):
    """Test generate_image_stability with default parameters."""
    mock_post, mock_response = mock_requests

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A simple robot",
            "model_id": "stability.stable-image-ultra-v1:1",
            "stability_api_key": "sk-test-key",
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Check the default parameters were used
    args, kwargs = mock_post.call_args
    data = kwargs["data"]

    # Defaults from the tool spec
    assert kwargs["headers"]["accept"] == "image/*"  # default return_type is "image"
    assert data["output_format"] == "png"  # default
    # seed=0 means random, so it shouldn't be in the data
    assert "seed" not in data

    assert result["status"] == "success"


def test_generate_image_stability_with_image_input(mock_requests):
    """Test generate_image_stability with image-to-image mode."""
    mock_post, mock_response = mock_requests

    # Create base64 encoded test image
    test_image_data = b"fake_image_data"
    test_image_b64 = base64.b64encode(test_image_data).decode("utf-8")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "Transform this image",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
            "mode": "image-to-image",
            "image": test_image_b64,
            "strength": 0.8,
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify the image was included in files
    args, kwargs = mock_post.call_args
    assert "image" in kwargs["files"]
    assert kwargs["files"]["image"] == test_image_data

    # Check that strength was included for image-to-image
    assert kwargs["data"]["strength"] == 0.8

    # Check that aspect_ratio was NOT included (only for text-to-image)
    assert "aspect_ratio" not in kwargs["data"]

    assert result["status"] == "success"


def test_generate_image_stability_with_data_url_image(mock_requests):
    """Test generate_image_stability with data URL image input."""
    mock_post, mock_response = mock_requests

    # Create data URL formatted image
    test_image_data = b"fake_image_data"
    test_image_b64 = base64.b64encode(test_image_data).decode("utf-8")
    data_url = f"data:image/png;base64,{test_image_b64}"

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "Transform this image",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
            "mode": "image-to-image",
            "image": data_url,
            "strength": 0.5,
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify the image was decoded correctly from data URL
    args, kwargs = mock_post.call_args
    assert "image" in kwargs["files"]
    assert kwargs["files"]["image"] == test_image_data

    assert result["status"] == "success"


def test_generate_image_stability_with_negative_prompt(mock_requests):
    """Test generate_image_stability with negative prompt."""
    mock_post, mock_response = mock_requests

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A beautiful landscape",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
            "negative_prompt": "blurry, low quality, distorted",
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Check that negative_prompt was included
    args, kwargs = mock_post.call_args
    assert kwargs["data"]["negative_prompt"] == "blurry, low quality, distorted"

    assert result["status"] == "success"


def test_generate_image_stability_seed_handling(mock_requests):
    """Test that seed=0 results in random seed (not included in request)."""
    mock_post, mock_response = mock_requests

    # Test with seed=0 (should not be included)
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A robot",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
            "seed": 0,
        },
    }

    generate_image_stability.generate_image_stability(tool=tool_use)
    args, kwargs = mock_post.call_args
    assert "seed" not in kwargs["data"]

    # Test with non-zero seed (should be included)
    tool_use["input"]["seed"] = 42
    generate_image_stability.generate_image_stability(tool=tool_use)
    args, kwargs = mock_post.call_args
    assert kwargs["data"]["seed"] == 42


def test_generate_image_stability_error_handling(mock_requests):
    """Test error handling in generate_image_stability."""
    mock_post, mock_response = mock_requests

    # Setup requests to raise an exception
    mock_response.raise_for_status.side_effect = Exception("API error")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "A robot",
            "model_id": "stability.stable-image-core-v1:1",
            "stability_api_key": "sk-test-key",
        },
    }

    result = generate_image_stability.generate_image_stability(tool=tool_use)

    # Verify error handling
    assert result["status"] == "error"
    assert "Error generating image: API error" in result["content"][0]["text"]


def test_call_stability_api_endpoint_routing():
    """Test that call_stability_api routes to correct endpoints."""
    from strands_tools.generate_image_stability import call_stability_api

    with patch("requests.post") as mock_post:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = b"mock_image_data"
        mock_response.headers = {"finish_reason": "SUCCESS"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Test each model endpoint
        test_cases = [
            ("stability.sd3-5-large-v1:0", "https://api.stability.ai/v2beta/stable-image/generate/sd3"),
            ("stability.stable-image-core-v1:1", "https://api.stability.ai/v2beta/stable-image/generate/core"),
            ("stability.stable-image-ultra-v1:1", "https://api.stability.ai/v2beta/stable-image/generate/ultra"),
        ]

        for model_id, expected_url in test_cases:
            # Reset mock
            mock_post.reset_mock()

            # Call the function
            image_data, finish_reason = call_stability_api(
                prompt="Test prompt", model_id=model_id, stability_api_key="sk-test-key", return_type="image"
            )

            # Verify the correct URL was called
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == expected_url

            # Verify the result
            assert image_data == b"mock_image_data"
            assert finish_reason == "SUCCESS"


def test_call_stability_api_direct():
    """Test the call_stability_api function directly."""
    from strands_tools.generate_image_stability import call_stability_api

    with patch("requests.post") as mock_post:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = b"mock_image_data"
        mock_response.headers = {"finish_reason": "SUCCESS"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Call the function
        image_data, finish_reason = call_stability_api(
            prompt="Test prompt",
            model_id="stability.stable-image-core-v1:1",
            stability_api_key="sk-test-key",
            return_type="image",
        )

        # Verify the result
        assert image_data == b"mock_image_data"
        assert finish_reason == "SUCCESS"

        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["authorization"] == "Bearer sk-test-key"
        assert kwargs["data"]["prompt"] == "Test prompt"


def test_generate_image_stability_via_agent(agent, mock_requests):
    """Test image generation via the agent interface."""
    mock_post, mock_response = mock_requests

    # This simulates how the tool would be used through the Agent interface
    result = agent.tool.generate_image_stability(
        prompt="Test via agent", model_id="stability.stable-image-core-v1:1", stability_api_key="sk-test-key"
    )

    result_text = extract_result_text(result)
    assert "Generated image using stability.stable-image-core-v1:1" in result_text
    assert "Finish reason: SUCCESS" in result_text
