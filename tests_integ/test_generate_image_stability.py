import base64
import os
import re
import shutil
from unittest import mock

import pytest
from strands import Agent
from strands_tools import generate_image_stability

"""
To run the tests, lease get an API key from the Stability AI platform:
https://platform.stability.ai/

Export it as an environment variable:
export STABILITY_API_KEY="your_api_key"
"""

if "STABILITY_API_KEY" not in os.environ:
    pytest.skip(allow_module_level=True, reason="STABILITY_API_KEY environment variable missing")


@pytest.fixture(autouse=True)
def os_environment():
    keys_to_copy = ["STABILITY_API_KEY", "STABILITY_MODEL_ID"]
    mock_env = {k: os.environ.get(k) for k in keys_to_copy if k in os.environ}
    with mock.patch.object(os, "environ", mock_env):
        yield mock_env


@pytest.fixture
def test_output_dir():
    path = "test_stability_output"
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


def test_stability_image_generation_integration():
    """
    End-to-end test for Stability AI image generation tool:
      - Verify API key is set
      - Generate an image with a unique prompt
      - Verify the response contains valid image data
      - Decode and validate the base64 image
      - Save the output image to a file
    """

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
    agent = Agent(tools=[generate_image_stability])

    # Test with empty prompt (should fail)
    result = agent.tool.generate_image_stability(prompt="")

    # The API should return an error for empty prompt
    assert result["status"] == "error"
    assert "Error generating image" in result["content"][0]["text"]


def test_stability_save_feature(os_environment, test_output_dir):
    """
    Test the image saving feature when STABILITY_OUTPUT_DIR is set.
    """
    # Create a test directory for output
    os_environment["STABILITY_OUTPUT_DIR"] = test_output_dir

    # Create agent with the tool
    agent = Agent(tools=[generate_image_stability])

    # Generate an image
    prompt = "A test image of a blue circle on a white background"
    result = agent.tool.generate_image_stability(prompt=prompt)

    # Verify success
    assert result["status"] == "success", f"Image generation failed: {result}"

    # Verify image content
    image_content = result["content"][1]
    assert "image" in image_content

    # Verify filename was included in the response
    text_content = result["content"][0]["text"]

    assert "saved to" in text_content.lower(), "Text shouldn't mention saving when disabled"
    filename = re.search(r"Image saved to (.+\.png)", text_content).group(1)

    # Verify the file exists
    assert os.path.exists(filename), f"File {filename} was not created"

    # Verify file contains image data
    file_size = os.path.getsize(filename)
    assert file_size > 1000, f"File {filename} seems too small to be a valid image ({file_size} bytes)"

    # Verify the text mentions the file was saved
    text_content = result["content"][0]["text"]
    assert "saved to" in text_content.lower(), "Text doesn't mention that image was saved"

    # Test that multiple images create different filenames
    result2 = agent.tool.generate_image_stability(prompt="A bright mural in a columbian town")
    assert result2["status"] == "success"

    filename2 = re.search(r"Image saved to (.+\.png)", result2["content"][0]["text"]).group(1)
    assert filename != filename2, "Multiple images should have different filenames"
    assert os.path.exists(filename2)

    # Verify filenames follow expected pattern
    filename_pattern = re.compile(r"\d{8}_\d{6}_[a-f0-9]{8}_[a-f0-9]{6}\.png$")
    assert filename_pattern.search(filename), f"Filename {filename} doesn't match expected pattern"


def test_stability_save_disabled():
    """
    Test that images are not saved when STABILITY_OUTPUT_DIR is not set.
    """

    # Create agent with the tool
    agent = Agent(tools=[generate_image_stability])

    # Generate an image
    result = agent.tool.generate_image_stability(prompt="A test image when saving is disabled")

    # Verify success
    assert result["status"] == "success", f"Image generation failed: {result}"

    # Verify image content exists
    image_content = result["content"][1]
    assert "image" in image_content

    print(">>>>>>>>>>>>>text content:", result["content"][0])

    # Verify the text doesn't mention saving image to disk
    text_content = result["content"][0]["text"]
    assert "saved to" not in text_content.lower(), "Text shouldn't mention saving when disabled"
