import base64
import os

import pytest
from strands import Agent
from strands_tools import generate_image_stability

"""
To run the tests, lease get an API key from the Stability AI platform:
https://platform.stability.ai/

Export it as an environment variable:
export STABILITY_API_KEY="your_api_key"
"""


def test_stability_image_generation_integration():
    """
    End-to-end test for Stability AI image generation tool:
      - Verify API key is set
      - Generate an image with a unique prompt
      - Verify the response contains valid image data
      - Decode and validate the base64 image
      - Save the output image to a file
    """
    # Skip if no API key is available
    if not os.environ.get("STABILITY_API_KEY"):
        pytest.skip("STABILITY_API_KEY not set - skipping integration test")

    # Create agent with the tool
    agent = Agent(tools=[generate_image_stability])

    # Create a unique prompt
    import uuid

    test_uuid = str(uuid.uuid4())[:8]
    unique_prompt = f"A figurine holding a sign with {test_uuid} written on it, photorealistic style"

    # Test 1: Basic image generation with defaults
    result = agent.tool.generate_image_stability(prompt=unique_prompt)

    # Verify success
    assert result["status"] == "success", f"Image generation failed: {result}"

    # Verify we have content
    assert "content" in result
    assert len(result["content"]) >= 2, "Expected text and image in content"

    # Verify text response
    text_content = result["content"][0]
    assert "text" in text_content
    assert "Generated image using" in text_content["text"]

    # Verify image data
    image_content = result["content"][1]
    assert "image" in image_content
    assert "format" in image_content["image"]
    assert "source" in image_content["image"]
    assert "bytes" in image_content["image"]["source"]

    # Verify the image bytes are valid
    image_bytes = image_content["image"]["source"]["bytes"]
    assert isinstance(image_bytes, bytes)
    assert len(image_bytes) > 1000, "Image seems too small to be valid"

    # Save the image to a file
    with open("output.png", "wb") as f:
        f.write(image_bytes)
    print(f"Saved test image to output.png with test ID: {test_uuid}")

    # Test 2: Generation with specific parameters
    result_custom = agent.tool.generate_image_stability(
        prompt="A beautiful sunset over mountains",
        aspect_ratio="16:9",
        style_preset="photographic",
        seed=42,  # Fixed seed for reproducibility
        negative_prompt="people, buildings, text",
    )

    assert result_custom["status"] == "success"

    # Optionally save the second test image as well
    custom_image_bytes = result_custom["content"][1]["image"]["source"]["bytes"]
    with open("output_novel_prompt.png", "wb") as f:
        f.write(custom_image_bytes)
    print("Saved custom test image to output_custom.png")

    # Test 3: Test with different model if specified
    if os.environ.get("STABILITY_MODEL_ID") == "stability.sd3-5-large-v1:0":
        # Test cfg_scale parameter (only works with SD3.5)
        result_sd35 = agent.tool.generate_image_stability(prompt="A futuristic cityscape", cfg_scale=8.0)
        assert result_sd35["status"] == "success"
        assert "stability.sd3-5-large-v1:0" in result_sd35["content"][0]["text"]

        # Save the SD3.5 test image as well
        sd35_image_bytes = result_sd35["content"][1]["image"]["source"]["bytes"]
        with open("output_sd35.png", "wb") as f:
            f.write(sd35_image_bytes)
        print("Saved SD3.5 test image to output_sd35.png")


def test_stability_image_to_image_integration():
    """
    Test image-to-image generation functionality.
    """
    if not os.environ.get("STABILITY_API_KEY"):
        pytest.skip("STABILITY_API_KEY not set - skipping integration test")

    agent = Agent(tools=[generate_image_stability])

    # Create a small test image (1x1 white pixel PNG)
    test_image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    test_image_b64 = base64.b64encode(test_image_bytes).decode("utf-8")

    # Test image-to-image transformation
    result = agent.tool.generate_image_stability(
        prompt="Transform this into a colorful abstract pattern",
        mode="image-to-image",
        image=test_image_b64,
        strength=0.8,
    )

    assert result["status"] == "success", f"Image-to-image generation failed: {result}"

    # Verify the output is different from input (due to transformation)
    output_bytes = result["content"][1]["image"]["source"]["bytes"]
    assert output_bytes != test_image_bytes, "Output should be different from input"


def test_stability_error_handling_integration():
    """
    Test error handling with invalid inputs.
    """
    if not os.environ.get("STABILITY_API_KEY"):
        pytest.skip("STABILITY_API_KEY not set - skipping integration test")

    agent = Agent(tools=[generate_image_stability])

    # Test with empty prompt (should fail)
    result = agent.tool.generate_image_stability(prompt="")

    # The API should return an error for empty prompt
    assert result["status"] == "error"
    assert "Error generating image" in result["content"][0]["text"]
