"""
Image generation tool for Strands Agent using Stability Platform API.

This module provides functionality to generate high-quality images using Stability AI's
latest models including SD3.5, Stable Image Ultra, and Stable Image Core through the
Stability Platform API.

Key Features:

1. Image Generation:
   • Text-to-image and image-to-image conversion
   • Support for multiple Stability AI models
   • Customizable generation parameters (seed, cfg_scale, aspect_ratio)
   • Style preset selection for consistent aesthetics
   • Flexible output formats (JPEG, PNG, WebP)

2. Response Format:
   • Rich response with both text and image data
   • Status tracking and error handling
   • Direct image data for immediate display

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import generate_image_stability

agent = Agent(tools=[generate_image_stability])

# Basic usage
agent.tool.generate_image_stability(
    prompt="A futuristic robot in a cyberpunk city",
    model_id="stability.stable-image-ultra-v1:1",
    stability_api_key="sk-xxx"
)

# Advanced usage with custom parameters
agent.tool.generate_image_stability(
    prompt="A serene mountain landscape",
    model_id="stability.sd3-5-large-v1:0",
    aspect_ratio="16:9",
    style_preset="photographic",
    cfg_scale=7.0,
    seed=42,
    stability_api_key="sk-xxx"
)
```
"""

import base64
from typing import Any, Tuple, Union

import requests
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "generate_image_stability",
    "description": (
        "Generates an image using Stability AI's latest models on the Stability Platform, "
        "including SD3.5, Stable Image Ultra and Stable Image Core."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "stability_api_key": {
                "type": "string",
                "description": (
                    "Your Stability API key for authentication. " "Required for accessing the Stability AI API."
                ),
            },
            "return_type": {
                "type": "string",
                "description": (
                    "The format in which to return the generated image. "
                    "Use 'image' to return the image data directly, or 'json' "
                    "to return a JSON object containing the image data as a base64-encoded string."
                ),
                "enum": ["json", "image"],
                "default": "image",
            },
            "prompt": {
                "type": "string",
                "description": "The text prompt to generate the image from",
            },
            "model_id": {
                "type": "string",
                "description": "The model to use for image generation.",
                "enum": [
                    "stability.stable-image-core-v1:1",
                    "stability.stable-image-ultra-v1:1",
                    "stability.sd3-5-large-v1:0",
                ],
            },
            "aspect_ratio": {
                "type": "string",
                "description": (
                    "Controls the aspect ratio of the generated image. "
                    "This parameter is only valid for text-to-image requests. "
                    "For image-to-image requests, the aspect ratio is determined by the input image."
                ),
                "enum": ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
                "default": "1:1",
            },
            "seed": {
                "type": "integer",
                "description": (
                    "Optional: Seed for random number generation. "
                    "Omit this parameter or pass 0 to use a random seed."
                ),
                "minimum": 0,
                "maximum": 4294967294,
                "default": 0,
            },
            "cfg_scale": {
                "type": "number",
                "description": (
                    "CFG scale for image generation (only used for stability.sd3-5-large-v1:0). "
                    "Controls how closely the image follows the prompt. "
                    "Higher values mean stricter adherence."
                ),
                "minimum": 1.0,
                "maximum": 10.0,
                "default": 4.0,
            },
            "output_format": {
                "type": "string",
                "description": "Output format for the generated image",
                "enum": ["jpeg", "png", "webp"],
            },
            "style_preset": {
                "type": "string",
                "description": (
                    "Style preset for image generation. " "Applies a predefined artistic style to the output"
                ),
                "enum": [
                    "3d-model",
                    "analog-film",
                    "anime",
                    "cinematic",
                    "comic-book",
                    "digital-art",
                    "enhance",
                    "fantasy-art",
                    "isometric",
                    "line-art",
                    "low-poly",
                    "modeling-compound",
                    "neon-punk",
                    "origami",
                    "photographic",
                    "pixel-art",
                    "tile-texture",
                ],
            },
            "image": {
                "type": "string",
                "description": (
                    "Input image for image-to-image generation. "
                    "Should be base64-encoded image data in jpeg, png or webp format."
                ),
            },
            "mode": {"type": "string", "description": "Mode of operation", "enum": ["text-to-image", "image-to-image"]},
            "strength": {
                "type": "number",
                "description": (
                    "Sometimes referred to as denoising, this parameter controls how much influence the "
                    "image parameter has on the generated image. A value of 0 would yield an image that "
                    "is identical to the input. A value of 1 would be as if you passed in no image at all. "
                    "Only used when mode is image-to-image."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "negative_prompt": {
                "type": "string",
                "description": (
                    "Text describing what you do not want to see in the generated image. "
                    "Helps exclude unwanted elements or styles."
                ),
                "maxLength": 10000,
            },
        },
        "required": ["stability_api_key", "prompt", "model_id"],
    },
}


def api_route(model_id: str) -> str:
    """
    Generate the API route based on the model ID.

    Args:
        model_id: The model identifier to generate the route for.

    Returns:
        str: The complete API route for the specified model.
    """
    route_map = {
        "stability.sd3-5-large-v1:0": "sd3",
        "stability.stable-image-ultra-v1:1": "ultra",
        "stability.stable-image-core-v1:1": "core",
    }
    base_url = "https://api.stability.ai/v2beta/stable-image"
    return f"{base_url}/generate/{route_map[model_id]}"


def call_stability_api(
    prompt: str,
    model_id: str,
    stability_api_key: str,
    return_type: str = "image",
    aspect_ratio: str = "1:1",
    cfg_scale: float = 4.0,
    seed: int = 0,
    output_format: str = "png",
    style_preset: str = None,
    image: str = None,
    mode: str = "text-to-image",
    strength: float = None,
    negative_prompt: str = None,
) -> Tuple[Union[bytes, str], str]:
    """
    Generate images using Stability Platform API.

    Args:
        prompt: Text prompt for image generation
        model_id: Model to use for generation
        stability_api_key: API key for Stability Platform
        return_type: Return format - "json" or "image"
        aspect_ratio: Aspect ratio for the output image
        cfg_scale: CFG scale for prompt adherence
        seed: Random seed for reproducible results
        output_format: Output format (jpeg, png, webp)
        style_preset: Style preset to apply
        image: Input image for image-to-image generation
        mode: Generation mode (text-to-image or image-to-image)
        strength: Influence of input image (for image-to-image)
        negative_prompt: Text describing what not to include in the image

    Returns:
        Tuple of (image_data, finish_reason)
        - image_data: bytes if return_type="image", base64 string if return_type="json"
        - finish_reason: string indicating completion status
    """
    # Determine API endpoint based on model
    model_endpoints = {
        "stability.sd3-5-large-v1:0": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        "stability.stable-image-core-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/core",
        "stability.stable-image-ultra-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
    }

    if model_id not in model_endpoints:
        raise ValueError(f"Unsupported model_id: {model_id}")

    url = model_endpoints[model_id]

    # Set accept header based on return type
    accept_header = "application/json" if return_type == "json" else "image/*"

    # Prepare headers
    headers = {"authorization": f"Bearer {stability_api_key}", "accept": accept_header}

    # Prepare data payload
    data = {
        "prompt": prompt,
        "output_format": output_format,
    }

    # Add optional parameters
    if aspect_ratio and mode == "text-to-image":
        data["aspect_ratio"] = aspect_ratio
    if cfg_scale is not None:
        data["cfg_scale"] = cfg_scale
    if seed is not None and seed > 0:
        data["seed"] = seed
    if style_preset:
        data["style_preset"] = style_preset
    if strength is not None and mode == "image-to-image":
        data["strength"] = strength
    if negative_prompt:
        data["negative_prompt"] = negative_prompt

    # Prepare files
    files = {}
    if image:
        # Handle base64 encoded image data
        if image.startswith("data:"):
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            image = image.split(",", 1)[1]

        # Decode base64 image data
        image_bytes = base64.b64decode(image)
        files["image"] = image_bytes
    else:
        files["none"] = ""

    # Make the API request
    response = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
    )

    response.raise_for_status()

    # Extract finish_reason and image data based on return type
    if return_type == "json":
        response_data = response.json()
        finish_reason = response_data.get("finish_reason", "SUCCESS")
        # Assuming the JSON response contains base64 image data
        image_data = response_data.get("image", "")
        return image_data, finish_reason
    else:
        finish_reason = response.headers.get("finish_reason", "SUCCESS")
        image_data = response.content
        return image_data, finish_reason


def generate_image_stability(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Generate images from text prompts using Stability Platform API.

    This function transforms textual descriptions into high-quality images using
    Stability AI's latest models including SD3.5, Stable Image Ultra, and Stable Image Core.
    It provides extensive customization options and handles the complete process from
    API interaction to image storage and result formatting.

    Generation Parameters:
    --------------------
    - prompt: The textual description of the desired image
    - model_id: Specific model to use (e.g., stability.stable-image-core-v1:1)
    - return_type: Format of returned image ("json" or "image", default: "image")
    - seed: Controls randomness for reproducible results (0 = random)
    - style_preset: Artistic style to apply (e.g., photographic, cinematic)
    - cfg_scale: Controls how closely the image follows the prompt, only used for stability.sd3-5-large-v1:0
    - aspect_ratio: Controls the aspect ratio of the output image
    - output_format: Output format (jpeg, png, webp)
    - mode: Generation mode (text-to-image or image-to-image)
    - strength: Influence of input image for image-to-image generation
    - negative_prompt: Text describing what not to include in the image
    - image: Input image for image-to-image generation

    Args:
        tool: ToolUse object containing the parameters for image generation.
            - prompt: The text prompt describing the desired image.
            - model_id: Model identifier (e.g., "stability.stable-image-core-v1:1").
            - stability_api_key: API key for Stability Platform authentication.
            - return_type: Optional return format ("json" or "image", default: "image").
            - seed: Optional random seed (default: 0 for random).
            - style_preset: Optional style preset name.
            - cfg_scale: Optional CFG scale value (default: 4.0).
            - aspect_ratio: Optional aspect ratio (default: "1:1").
            - output_format: Optional output format (default: "png").
            - mode: Optional generation mode (default: "text-to-image").
            - strength: Optional input image influence (default: varies by mode).
            - negative_prompt: Optional negative prompt to exclude elements.
            - image: Optional input image for image-to-image generation.

        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult: A dictionary containing the result status and content:
            - On success: Contains a text message and the rendered image data.
            - On failure: Contains an error message describing what went wrong.

    Notes:
        - The function requires a Stability AI API key for authentication. See https://platform.stability.ai/
        - For best results, provide detailed, descriptive prompts
        - Returns a ToolResult with image data and status information
    """
    try:
        tool_input = tool.get("input", tool)
        tool_use_id = tool.get("toolUseId", "default_id")

        # Extract input parameters with defaults
        prompt = tool_input.get("prompt")
        model_id = tool_input.get("model_id")
        stability_api_key = tool_input.get("stability_api_key")
        return_type = tool_input.get("return_type", "image")
        aspect_ratio = tool_input.get("aspect_ratio", "1:1")
        cfg_scale = tool_input.get("cfg_scale", 4.0)
        seed = tool_input.get("seed", 0)
        output_format = tool_input.get("output_format", "png")
        style_preset = tool_input.get("style_preset")
        image = tool_input.get("image")
        mode = tool_input.get("mode", "text-to-image")
        strength = tool_input.get("strength")
        negative_prompt = tool_input.get("negative_prompt")

        # Generate the image using the API
        image_data, finish_reason = call_stability_api(
            prompt=prompt,
            model_id=model_id,
            stability_api_key=stability_api_key,
            return_type=return_type,
            aspect_ratio=aspect_ratio,
            cfg_scale=cfg_scale,
            seed=seed,
            output_format=output_format,
            style_preset=style_preset,
            image=image,
            mode=mode,
            strength=strength,
            negative_prompt=negative_prompt,
        )

        # Handle image data based on return type
        if return_type == "json":
            # image_data is base64 string - decode it for the ToolResult
            image_bytes = base64.b64decode(image_data)
        else:
            # image_data is already bytes
            image_bytes = image_data

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"Generated image using {model_id}. Finish reason: {finish_reason}"},
                {
                    "image": {
                        "format": output_format,
                        "source": {"bytes": image_bytes},
                    }
                },
            ],
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error generating image: {str(e)}"}],
        }
