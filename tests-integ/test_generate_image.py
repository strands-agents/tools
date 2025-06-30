import os

import pytest
from strands import Agent
from strands_tools import generate_image, image_reader


@pytest.fixture
def agent():
    """Agent with image generation and reader tools."""
    return Agent(tools=[generate_image, image_reader])


def test_generate_and_read_image(agent, tmp_path):
    # 1. Generate a lovely dog picture
    prompt = "A corgi riding a skateboard in Times Square"
    image_gen_result = agent.tool.generate_image(
        prompt=prompt,
        model_id="stability.stable-image-core-v1:1",
        aspect_ratio="1:1",
        output_format="png",
        negative_prompt="blurry, low quality",
    )
    assert image_gen_result["status"] == "success", str(image_gen_result)
    content = image_gen_result["content"]

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
    image_path = tmp_path / "generated.png"
    with open(image_path, "wb") as f:
        f.write(found_image)

    # 2. use image_reader tool to verify it's a real image
    assert os.path.exists(image_path), f"Image file not found at {image_path}"
    read_result = agent.tool.image_reader(image_path=str(image_path))
    assert read_result["status"] == "success", str(read_result)
    image_content = read_result["content"][0]["image"]
    assert image_content["format"] == "png"
    assert isinstance(image_content["source"]["bytes"], bytes)
    assert len(image_content["source"]["bytes"]) > 1000

    # 3. test semantic usage to check if it recognizes dog/corgi
    semantic_result = agent(f"What is the image at `{image_path}`")
    assert "dog" in str(semantic_result).lower() or "corgi" in str(semantic_result).lower()
