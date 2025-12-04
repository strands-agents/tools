"""
Integration tests for the generate_image_gemini tool.

These tests require a valid GOOGLE_API_KEY environment variable to be set.
They make real API calls to Google Gemini and verify the complete workflow.
"""

import os

import pytest
from strands import Agent

from strands_tools import generate_image_gemini, image_reader


@pytest.fixture
def agent():
    """Agent with Gemini image generation and reader tools."""
    return Agent(tools=[generate_image_gemini, image_reader])


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_generate_image_basic(agent, tmp_path):
    """Test basic image generation with default parameters."""
    prompt = "A robot holding a red skateboard"

    # Generate image
    result = agent.tool.generate_image_gemini(prompt=prompt)

    # Verify success
    assert result["status"] == "success", str(result)
    content = result["content"]

    # Extract and verify image bytes from result
    found_image = None
    for item in content:
        if "image" in item and "source" in item["image"]:
            found_image = item["image"]["source"]["bytes"]
            assert isinstance(found_image, bytes), "Returned image bytes are not 'bytes' type"
            assert len(found_image) > 1000, "Returned image is too small to be valid"
            break
    assert found_image is not None, "No image bytes found in result"

    # Save image to temp directory
    image_path = tmp_path / "generated_robot.png"
    with open(image_path, "wb") as f:
        f.write(found_image)

    # Verify the file was created
    assert os.path.exists(image_path), f"Image file not found at {image_path}"
    assert os.path.getsize(image_path) > 1000, "Generated image file is too small"


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_generate_image_with_aspect_ratio(agent, tmp_path):
    """Test image generation with custom aspect ratio."""
    prompt = "A serene mountain landscape at sunset"

    # Generate image with 16:9 aspect ratio
    result = agent.tool.generate_image_gemini(
        prompt=prompt,
        aspect_ratio="16:9",
    )

    # Verify success
    assert result["status"] == "success", str(result)
    content = result["content"]

    # Extract image bytes
    found_image = None
    for item in content:
        if "image" in item and "source" in item["image"]:
            found_image = item["image"]["source"]["bytes"]
            break
    assert found_image is not None, "No image bytes found in result"

    # Save and verify
    image_path = tmp_path / "landscape_16_9.png"
    with open(image_path, "wb") as f:
        f.write(found_image)

    assert os.path.exists(image_path)
    assert os.path.getsize(image_path) > 1000


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_generate_and_read_image(agent, tmp_path):
    """Test complete workflow: generate image and read it back."""
    prompt = "A cute corgi puppy playing in a park"

    # 1. Generate image
    image_gen_result = agent.tool.generate_image_gemini(
        prompt=prompt,
        aspect_ratio="1:1",
    )
    assert image_gen_result["status"] == "success", str(image_gen_result)
    content = image_gen_result["content"]

    # Extract and verify image bytes
    found_image = None
    for item in content:
        if "image" in item and "source" in item["image"]:
            found_image = item["image"]["source"]["bytes"]
            assert isinstance(found_image, bytes)
            assert len(found_image) > 1000
            break
    assert found_image is not None

    # Save image to temp directory
    image_path = tmp_path / "corgi.png"
    with open(image_path, "wb") as f:
        f.write(found_image)

    # 2. Use image_reader tool to verify it's a real image
    assert os.path.exists(image_path), f"Image file not found at {image_path}"
    read_result = agent.tool.image_reader(image_path=str(image_path))
    assert read_result["status"] == "success", str(read_result)
    image_content = read_result["content"][0]["image"]
    # Gemini may return jpeg or png format depending on the model
    assert image_content["format"] in ["png", "jpeg"], f"Unexpected format: {image_content['format']}"
    assert isinstance(image_content["source"]["bytes"], bytes)
    assert len(image_content["source"]["bytes"]) > 1000

    # 3. Test semantic usage to check if it recognizes the subject (optional - requires AWS credentials)
    try:
        semantic_result = agent(f"What is in the image at `{image_path}`?")
        result_text = str(semantic_result).lower()
        # If semantic analysis works, verify it recognizes the subject
        assert "dog" in result_text or "corgi" in result_text or "puppy" in result_text
    except Exception as e:
        # Skip semantic test if AWS credentials are not available
        if "security token" in str(e).lower() or "credentials" in str(e).lower():
            pytest.skip(f"Skipping semantic test - AWS credentials not available: {e}")
        else:
            raise


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_generate_image_with_different_model(agent, tmp_path):
    """Test image generation with a different Gemini model."""
    prompt = "A futuristic cityscape with flying cars"

    # Generate image with gemini-2.5-flash-image model
    result = agent.tool.generate_image_gemini(
        prompt=prompt,
        model_id="gemini-2.5-flash-image",
        aspect_ratio="21:9",
    )

    # Verify success
    assert result["status"] == "success", str(result)
    content = result["content"]

    # Extract image bytes
    found_image = None
    for item in content:
        if "image" in item and "source" in item["image"]:
            found_image = item["image"]["source"]["bytes"]
            break
    assert found_image is not None

    # Save and verify
    image_path = tmp_path / "cityscape_21_9.png"
    with open(image_path, "wb") as f:
        f.write(found_image)

    assert os.path.exists(image_path)
    assert os.path.getsize(image_path) > 1000


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)
def test_generate_image_various_aspect_ratios(agent, tmp_path):
    """Test image generation with various aspect ratios."""
    prompt = "A simple geometric pattern"
    aspect_ratios = ["1:1", "4:3", "16:9", "9:16"]

    for ratio in aspect_ratios:
        result = agent.tool.generate_image_gemini(
            prompt=prompt,
            aspect_ratio=ratio,
        )

        # Verify success
        assert result["status"] == "success", f"Failed for aspect ratio {ratio}: {result}"

        # Extract image bytes
        found_image = None
        for item in result["content"]:
            if "image" in item and "source" in item["image"]:
                found_image = item["image"]["source"]["bytes"]
                break
        assert found_image is not None, f"No image found for aspect ratio {ratio}"

        # Save and verify
        safe_ratio = ratio.replace(":", "_")
        image_path = tmp_path / f"pattern_{safe_ratio}.png"
        with open(image_path, "wb") as f:
            f.write(found_image)

        assert os.path.exists(image_path)
        assert os.path.getsize(image_path) > 1000


def test_generate_image_missing_api_key(agent):
    """Test error handling when API key is missing."""
    # Temporarily remove API key
    original_key = os.environ.pop("GOOGLE_API_KEY", None)

    try:
        result = agent.tool.generate_image_gemini(prompt="A test image")

        # Verify error response
        assert result["status"] == "error"
        assert "GOOGLE_API_KEY" in result["content"][0]["text"]
    finally:
        # Restore API key if it existed
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key


def test_generate_image_invalid_aspect_ratio(agent):
    """Test error handling for invalid aspect ratio."""
    result = agent.tool.generate_image_gemini(
        prompt="A test image",
        aspect_ratio="5:5",  # Invalid ratio
    )

    # Verify error response
    assert result["status"] == "error"
    assert "Invalid aspect_ratio" in result["content"][0]["text"]


def test_generate_image_invalid_model(agent):
    """Test error handling for invalid model ID."""
    result = agent.tool.generate_image_gemini(
        prompt="A test image",
        model_id="invalid-model-id",
    )

    # Verify error response
    assert result["status"] == "error"
    assert "Invalid model_id" in result["content"][0]["text"]
