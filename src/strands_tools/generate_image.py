"""
Image generation tool for Strands Agent using Amazon Bedrock.

This module provides functionality to generate high-quality images using Amazon Bedrock's
text-to-image models based on text prompts. It handles the entire image generation
process including API integration, parameter management, response processing, and
local storage of results.

Key Features:

1. Image Generation:
   • Text-to-image conversion using various Bedrock models
   • Support for multiple model families (Amazon Titan, Stability AI)
   • Dynamic model selection when no model is specified
   • Customizable generation parameters (seed, cfg_scale)

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
    model_id="amazon.titan-image-generator-v2:0",
    region="us-west-2",
    seed=42,
    cfg_scale=12
)
```

See the generate_image function docstring for more details on parameters and options.
"""

import base64
import json
import os
import re
from typing import Any, List, Tuple

import boto3
from botocore.exceptions import ClientError
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "generate_image",
    "description": "Generates an image using Amazon Bedrock models based on a given prompt",
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
                    "description": "Model ID for image generation (e.g., amazon.titan-image-generator-v2:0, "
                                 "stability.stable-image-core-v1:1)",
                },
                "region": {
                    "type": "string",
                    "description": "Optional: AWS region to use (default: from AWS_REGION env variable or us-west-2)",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional: Seed for deterministic generation (default: 2147483646)",
                },
                "cfg_scale": {
                    "type": "number",
                    "description": "Optional: CFG scale for image generation (default: 10)",
                },
            },
            "required": ["prompt"],
        }
    },
}


def validate_model_in_region(model_id: str, region: str) -> Tuple[bool, List[str]]:
    """
    Validate if the specified model is available in the given region using the Bedrock API.

    This function checks if the model supports text-to-image generation capability in the specified region
    by examining the model's inputModalities and outputModalities. It also checks if the model
    is in LEGACY status or doesn't support ON_DEMAND inference, in which case it raises an exception.

    Args:
        model_id: The model ID to validate
        region: The AWS region to check against

    Returns:
        tuple: (is_valid, available_models)
            - is_valid: True if the model is supported in the region, False otherwise
            - available_models: List of available text-to-image models in the region

    Raises:
        ValueError: If the model is in LEGACY status or doesn't support ON_DEMAND inference
    """
    try:
        # Create a Bedrock client to list available models
        bedrock_client = boto3.client("bedrock", region_name=region)

        # Get list of foundation models available in the region
        response = bedrock_client.list_foundation_models()

        # Filter for text-to-image models based on input and output modalities
        available_models = []
        legacy_models = []
        non_on_demand_models = []

        for model in response.get("modelSummaries", []):
            model_id_from_api = model.get("modelId", "")
            input_modalities = model.get("inputModalities", [])
            output_modalities = model.get("outputModalities", [])

            # Check if this model supports text-to-image generation
            # It should take TEXT as input and produce IMAGE as output
            if "TEXT" in input_modalities and "IMAGE" in output_modalities:
                # Check if the model is in LEGACY status
                model_lifecycle = model.get("modelLifecycle", {})
                status = model_lifecycle.get("status", "")

                # Check if the model supports ON_DEMAND inference
                inference_types = model.get("inferenceTypesSupported", [])
                supports_on_demand = "ON_DEMAND" in inference_types

                if status == "LEGACY":
                    legacy_models.append(model_id_from_api)
                elif not supports_on_demand:
                    non_on_demand_models.append(model_id_from_api)
                else:
                    available_models.append(model_id_from_api)

        # Check if the requested model is in the list of legacy models
        if any(model_id == legacy_model for legacy_model in legacy_models):
            raise ValueError(
                f"Model '{model_id}' is in LEGACY status and no longer recommended for use. "
                f"Please use one of the active models instead: {', '.join(available_models)}"
            )

        # Check if the requested model is in the list of non-on-demand models
        if any(model_id == non_on_demand_model for non_on_demand_model in non_on_demand_models):
            raise ValueError(
                f"Model '{model_id}' does not support on-demand throughput. "
                f"Please use one of these models that support on-demand inference: {', '.join(available_models)}"
            )

        # Check if the requested model is in the list of available models
        is_valid = any(model_id == available_model for available_model in available_models)

        return is_valid, available_models

    except ValueError as e:
        # Re-raise ValueError for legacy models or non-on-demand models
        raise e
    except Exception as e:
        # If we can't access the API, return False and empty list
        print(f"Error checking model availability: {str(e)}")
        return False, []


def create_filename(prompt: str) -> str:
    """
    Generate a filename from the prompt text.

    Args:
        prompt: The text prompt used for image generation

    Returns:
        A sanitized filename based on the first few words of the prompt
    """
    words = re.findall(r"\w+", prompt.lower())[:5]
    filename = "_".join(words)
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    return filename[:100]  # Limit filename length


def generate_image(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Generate images from text prompts using Amazon Bedrock models.

    This function transforms textual descriptions into high-quality images using
    various text-to-image models available through Amazon Bedrock. It provides extensive
    customization options and handles the complete process from API interaction to
    image storage and result formatting.

    How It Works:
    ------------
    1. Extracts and validates parameters from the tool input
    2. Dynamically selects an available model if none is specified
    3. Configures the request payload with appropriate parameters for the model family
    4. Invokes the Bedrock image generation model through AWS SDK
    5. Processes the response to extract the base64-encoded image
    6. Creates an appropriate filename based on the prompt content
    7. Saves the image to a local output directory
    8. Returns a success response with both text description and rendered image

    Generation Parameters:
    --------------------
    - prompt: The textual description of the desired image
    - model_id: Specific model to use (if not provided, automatically selects one)
    - region: AWS region to use (defaults to AWS_REGION env variable or us-west-2)
    - seed: Controls randomness for reproducible results
    - cfg_scale: Controls how closely the image follows the prompt

    Supported Model Families:
    ----------------------
    - Amazon Titan Image Generator (v1 and v2)
    - Amazon Nova Canvas
    - Stability AI Stable Image (Core and Ultra)
    - Stability AI SD3

    Args:
        tool: ToolUse object containing the parameters for image generation.
            - prompt: The text prompt describing the desired image.
            - model_id: Optional model identifier.
            - region: Optional AWS region (default: from AWS_REGION env variable or us-west-2).
            - seed: Optional seed value (default: 2147483646).
            - cfg_scale: Optional CFG scale value (default: 10).
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

        # Get region from input, environment variable, or default to us-west-2
        region = tool_input.get("region", os.environ.get("AWS_REGION", "us-west-2"))

        # Check if model_id is explicitly provided
        if "model_id" in tool_input:
            model_id = tool_input["model_id"]
        else:
            # Find valid models in the region
            try:
                _, available_models = validate_model_in_region("", region)

                if not available_models:
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [
                            {
                                "text": f"No text-to-image models available in region '{region}'. "
                                       f"Please try a different region."
                            }
                        ],
                    }

                # Simply use the first available model
                model_id = available_models[0]
                print(f"No model_id provided. Using automatically selected model: {model_id}")
            except Exception as e:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [
                        {
                            "text": f"Error determining available models in region '{region}': {str(e)}. "
                                   f"Please specify a model_id explicitly."
                        }
                    ],
                }

        # Get seed from input or use a default value that works for all models
        seed = tool_input.get("seed", 2147483646)
        cfg_scale = tool_input.get("cfg_scale", 10)

        # Validate if the model is available in the specified region
        try:
            is_valid, available_models = validate_model_in_region(model_id, region)
            if not is_valid:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [
                        {
                            "text": f"Model '{model_id}' is not available in region '{region}'. "
                                   f"Available text-to-image models in this region include: "
                                   f"{', '.join(available_models)}"
                        }
                    ],
                }
        except ValueError as e:
            # Handle legacy model error
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": str(e)}],
            }
        except Exception:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [
                    {
                        "text": f"Could not validate model availability in region '{region}'. "
                               f"Please check your AWS credentials and permissions."
                    }
                ],
            }

        # Create a Bedrock Runtime client with the specified region
        try:
            client = boto3.client("bedrock-runtime", region_name=region)
        except ClientError as e:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Failed to create Bedrock client in region '{region}': {str(e)}"}],
            }

        # Format the request payload based on the model family
        if "stability" in model_id:
            # Format for Stability AI models (stable-image, sd3)
            native_request = {"prompt": prompt, "seed": seed}
        elif "amazon" in model_id:
            # Format for Amazon models (Titan, Nova Canvas)
            native_request = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "cfgScale": cfg_scale,
                    "seed": seed,
                },
            }
        else:
            # This should not happen due to the validation above, but keeping as a fallback
            raise ValueError(f"Unsupported model: {model_id}. Please use one of the supported image generation models.")

        request = json.dumps(native_request)

        # Invoke the model
        try:
            response = client.invoke_model(modelId=model_id, body=request)
        except ClientError as e:
            if "AccessDeniedException" in str(e):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [
                        {
                            "text": f"Access denied for model '{model_id}' in region '{region}'. "
                                   f"Please check your AWS credentials and permissions."
                        }
                    ],
                }
            elif "ValidationException" in str(e) and "not found" in str(e).lower():
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [
                        {
                            "text": f"Model '{model_id}' not found in region '{region}'. "
                                   f"Please verify the model ID and region."
                        }
                    ],
                }
            else:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Error invoking model: {str(e)}"}],
                }

        # Decode the response body
        model_response = json.loads(response["body"].read())

        # Extract the image data - handle different response formats
        try:
            if "stability" in model_id:
                # For Stability AI models
                base64_image_data = model_response["images"][0]
            elif "amazon" in model_id:
                # For Amazon Titan and Nova Canvas models
                if "images" in model_response and isinstance(model_response["images"], list):
                    if isinstance(model_response["images"][0], dict) and "imageBase64" in model_response["images"][0]:
                        base64_image_data = model_response["images"][0]["imageBase64"]
                    elif isinstance(model_response["images"][0], str):
                        base64_image_data = model_response["images"][0]
                    else:
                        raise ValueError("Unexpected Amazon model response format")
                else:
                    raise ValueError("Unexpected Amazon model response structure")
            else:
                raise ValueError(f"Unsupported model family: {model_id}")
        except (KeyError, IndexError) as e:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [
                    {
                        "text": f"Failed to extract image data from model response: {str(e)}. "
                               f"Response structure: {json.dumps(model_response, indent=2)[:500]}..."
                    }
                ],
            }

        # Create a filename based on the prompt
        filename = create_filename(prompt)

        # Save the generated image to a local folder
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Handle duplicate filenames
        i = 1
        base_image_path = os.path.join(output_dir, f"{filename}.png")
        image_path = base_image_path
        while os.path.exists(image_path):
            image_path = os.path.join(output_dir, f"{filename}_{i}.png")
            i += 1

        # Save the image to disk
        with open(image_path, "wb") as file:
            file.write(base64.b64decode(base64_image_data))

        # Return success response with image
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
