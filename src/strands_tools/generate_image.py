"""
Image generation tool for Strands Agent using Stable Diffusion.

This module provides functionality to generate high-quality images using Amazon Bedrock's
Stable Diffusion models based on text prompts. It handles the entire image generation
process including API integration, parameter management, response processing, and
local storage of results.

Key Features:

1. Image Generation:
   • Text-to-image conversion using Stable Diffusion
   • Support for multiple model variants (primarily stable-diffusion-xl-v1)
   • Customizable generation parameters (seed, steps, cfg_scale)
   • Style preset selection for consistent aesthetics

2. Output Management:
   • Automatic local saving with intelligent filename generation
   • Base64 encoding/decoding for transmission
   • Duplicate filename detection and resolution
   • Organized output directory structure

3. Response Format:
   • Rich response with both text and image data
   • Status tracking and error handling
   • Direct base64 image data for immediate display
   • File path reference for local access

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import generate_image

agent = Agent(tools=[generate_image])

# Basic usage with default parameters
agent.tool.generate_image(prompt="A steampunk robot playing chess")

# Advanced usage with custom parameters
agent.tool.generate_image(
    prompt="A futuristic city with flying cars",
    model_id="stability.stable-diffusion-xl-v1",
    seed=42,
    steps=50,
    cfg_scale=12,
    style_preset="cinematic"
)
```

See the generate_image function docstring for more details on parameters and options.
"""

import base64
import json
import os
import random
import re
from typing import Any

import boto3
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "generate_image",
    "description": "Generates an image using Stable Diffusion based on a given prompt",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt for image generation",
                },
                "model_id": {
                    "type": "string",
                    "description": "Model id for image model, stability.stable-diffusion-xl-v1.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional: Seed for random number generation (default: random)",
                },
                "steps": {
                    "type": "integer",
                    "description": "Optional: Number of steps for image generation (default: 30)",
                },
                "cfg_scale": {
                    "type": "number",
                    "description": "Optional: CFG scale for image generation (default: 10)",
                },
                "style_preset": {
                    "type": "string",
                    "description": "Optional: Style preset for image generation (default: 'photographic')",
                },
            },
            "required": ["prompt"],
        }
    },
}


def generate_image(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Generate images from text prompts using Stable Diffusion via Amazon Bedrock.

    This function transforms textual descriptions into high-quality images using
    Stable Diffusion models available through Amazon Bedrock. It provides extensive
    customization options and handles the complete process from API interaction to
    image storage and result formatting.

    How It Works:
    ------------
    1. Extracts and validates parameters from the tool input
    2. Configures the request payload with appropriate parameters
    3. Invokes the Bedrock image generation model through AWS SDK
    4. Processes the response to extract the base64-encoded image
    5. Creates an appropriate filename based on the prompt content
    6. Saves the image to a local output directory
    7. Returns a success response with both text description and rendered image

    Generation Parameters:
    --------------------
    - prompt: The textual description of the desired image
    - model_id: Specific model to use (defaults to stable-diffusion-xl-v1)
    - seed: Controls randomness for reproducible results
    - style_preset: Artistic style to apply (e.g., photographic, cinematic)
    - cfg_scale: Controls how closely the image follows the prompt
    - steps: Number of diffusion steps (higher = more refined but slower)

    Common Usage Scenarios:
    ---------------------
    - Creating illustrations for documents or presentations
    - Generating visual concepts for design projects
    - Visualizing scenes or characters for creative writing
    - Producing custom artwork based on specific descriptions
    - Testing visual ideas before commissioning real artwork

    Args:
        tool: ToolUse object containing the parameters for image generation.
            - prompt: The text prompt describing the desired image.
            - model_id: Optional model identifier (default: "stability.stable-diffusion-xl-v1").
            - seed: Optional random seed (default: random integer).
            - style_preset: Optional style preset name (default: "photographic").
            - cfg_scale: Optional CFG scale value (default: 10).
            - steps: Optional number of diffusion steps (default: 30).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult: A dictionary containing the result status and content:
            - On success: Contains a text message with the saved image path and the
              rendered image in base64 format.
            - On failure: Contains an error message describing what went wrong.

    Notes:
        - Image files are saved to an "output" directory in the current working directory
        - Filenames are generated based on the first few words of the prompt
        - Duplicate filenames are handled by appending an incrementing number
        - The function requires AWS credentials with Bedrock permissions
        - For best results, provide detailed, descriptive prompts
    """
    try:
        tool_use_id = tool["toolUseId"]
        tool_input = tool["input"]

        # Extract input parameters
        prompt = tool_input.get("prompt", "A stylized picture of a cute old steampunk robot.")
        model_id = tool_input.get("model_id", "stability.stable-diffusion-xl-v1")
        seed = tool_input.get("seed", random.randint(0, 4294967295))
        style_preset = tool_input.get("style_preset", "photographic")
        cfg_scale = tool_input.get("cfg_scale", 10)
        steps = tool_input.get("steps", 30)

        # Create a Bedrock Runtime client
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Format the request payload
        native_request = {
            "text_prompts": [{"text": prompt}],
            "style_preset": style_preset,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "steps": steps,
        }
        request = json.dumps(native_request)

        # Invoke the model
        response = client.invoke_model(modelId=model_id, body=request)

        # Decode the response body
        model_response = json.loads(response["body"].read())

        # Extract the image data
        base64_image_data = model_response["artifacts"][0]["base64"]

        # Create a filename based on the prompt
        def create_filename(prompt: str) -> str:
            """Generate a filename from the prompt text."""
            words = re.findall(r"\w+", prompt.lower())[:5]
            filename = "_".join(words)
            filename = re.sub(r"[^\w\-_\.]", "_", filename)
            return filename[:100]  # Limit filename length

        filename = create_filename(prompt)

        # Save the generated image to a local folder
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = 1
        base_image_path = os.path.join(output_dir, f"{filename}.png")
        image_path = base_image_path
        while os.path.exists(image_path):
            image_path = os.path.join(output_dir, f"{filename}_{i}.png")
            i += 1

        with open(image_path, "wb") as file:
            file.write(base64.b64decode(base64_image_data))

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"The generated image has been saved locally to {image_path}. "},
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": base64.b64decode(base64_image_data)},
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
