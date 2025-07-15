import os

import pytest
from strands import Agent
from strands_tools import nova_canvas, image_reader


@pytest.fixture
def agent():
    """Agent with image generation and reader tools."""
    return Agent(tools=[nova_canvas, image_reader])


def test_generate_and_read_image(agent, tmp_path):
    # 1. Generate a lovely dog picture
    prompt = "A corgi riding a skateboard in Times Square"
    image_gen_result = agent.tool.nova_canvas(
        prompt=prompt,
        task_type="TEXT_IMAGE",
        model_id="amazon.nova-canvas-v1:0",
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

def test_remove_background(agent, tmp_path):
    # 1. Generate an image
    prompt = "A corgi riding a skateboard in Times Square"
    image_gen_result = agent.tool.nova_canvas(
        prompt=prompt,
        task_type="TEXT_IMAGE",
        model_id="amazon.nova-canvas-v1:0",
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

    #2. Remove the background from the generated image
    image_gen_result = agent.tool.nova_canvas(
        task_type="BACKGROUND_REMOVAL",
        model_id="amazon.nova-canvas-v1:0",
        image_path=str(image_path),
    )
    assert image_gen_result["status"] == "success", str(image_gen_result)
    content = image_gen_result["content"]

    # Extract and verify the image with removed background bytes from result
    found_image = None
    for item in content:
        if "image" in item and "source" in item["image"]:
            found_image = item["image"]["source"]["bytes"]
            assert isinstance(found_image, bytes), "Returned image bytes are not 'bytes' type"
            assert len(found_image) > 1000, "Returned image is too small to be valid"
            break
    assert found_image is not None, "No image bytes found in result"

    # Save image to temp directory
    image_path_no_bg = tmp_path / "generated_no_bg.png"
    with open(image_path_no_bg, "wb") as f:
        f.write(found_image)

    # 2. use image_reader tool to verify it's a real image
    assert os.path.exists(image_path_no_bg), f"Image file not found at {image_path_no_bg}"
    read_result = agent.tool.image_reader(image_path=str(image_path_no_bg))
    assert read_result["status"] == "success", str(read_result)
    image_content = read_result["content"][0]["image"]
    assert image_content["format"] == "png"
    assert isinstance(image_content["source"]["bytes"], bytes)
    assert len(image_content["source"]["bytes"]) > 1000

    # 3. test semantic usage to check if it recognizes dog/corgi
    semantic_result = agent(f"Has the background been removed from the image at `{image_path_no_bg} - compare with image at {image_path}` \
                            respond with yes or no first")
    print(f"Agent response: {semantic_result}")
    assert "yes" in str(semantic_result).lower()