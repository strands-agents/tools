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
    assert "yes" in str(semantic_result).lower()

def test_virtual_try_on_mask_garment(agent, tmp_path):
    # 1. Generate an image of an empty living room
    prompt = "full body person with a warm, genuine smile  standing facing directly at the camera. \
            in a sunny neighberhood with green nature."
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

    # Save the empty living room with red couch image to temp directory
    living_room_image_path = tmp_path / "human_standing.png"
    with open(living_room_image_path, "wb") as f:
        f.write(found_image)

    # 2. Generate an image of a yellow couch
    prompt = "Generate a vibrant tech hoodie with AWS written on it"
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

    # Save the couch image to temp directory
    couch_image_path = tmp_path / "ai_hoodie_aws.png"
    with open(couch_image_path, "wb") as f:
        f.write(found_image)

    # 3. Virtual try on the couch on the empty living room
    image_gen_result = agent.tool.nova_canvas(
        task_type="VIRTUAL_TRY_ON",
        model_id="amazon.nova-canvas-v1:0",
        image_path=str(living_room_image_path),
        reference_image_path=str(couch_image_path),
        mask_type="GARMENT",
        garment_class="UPPER_BODY",
        longSleeveStyle="SLEEVE_DOWN"
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
    image_path = tmp_path / "hoodie_ai_garmet_try_on.png"
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
    semantic_result = agent(f"Does the image at path `{image_path}` contain a a person wearing an AWS hoodie?\
                            respond with yes or no first")
    assert "yes" in str(semantic_result).lower() 

# def test_virtual_try_on_prompt_mask(agent, tmp_path):
    # 1. Generate an image of an empty living room
    prompt = "a living room with a white background and a purple couch in the middle"
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

    # Save the empty living room with red couch image to temp directory
    living_room_image_path = tmp_path / "empty_room_blue_couch.png"
    with open(living_room_image_path, "wb") as f:
        f.write(found_image)

    # 2. Generate an image of a yellow couch
    prompt = "Generate a green couch with white background"
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

    # Save the couch image to temp directory
    couch_image_path = tmp_path / "couch.png"
    with open(couch_image_path, "wb") as f:
        f.write(found_image)

    # 3. Virtual try on the couch on the empty living room
    image_gen_result = agent.tool.nova_canvas(
        task_type="VIRTUAL_TRY_ON",
        model_id="amazon.nova-canvas-v1:0",
        image_path=str(living_room_image_path),
        reference_image_path=str(couch_image_path),
        mask_type="PROMPT",
        mask_prompt="replace the couch with yellow couch"
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
    image_path = tmp_path / "living_room_couch_try_on.png"
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
    semantic_result = agent(f"Does the image at path `{image_path}` contain a green couch in an empty living room?\
                            respond with yes or no first")
    print(f"\n Agent response: {semantic_result}")
    assert "yes" in str(semantic_result).lower() and "green" in str(semantic_result).lower()