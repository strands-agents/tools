"""
Image generation tool for Strands Agent using Google Gemini models.

This module provides functionality to generate high-quality images using Google's
Gemini API with native image generation capabilities. It handles the entire image
generation process including API integration, parameter management, response processing,
and local storage of results.

Key Features:

1. Image Generation:
   • Text-to-image conversion using Gemini models with image generation capability
   • Support for multiple model variants:
        • gemini-2.5-flash-image (Imagen 3 Fast)
        • gemini-3-pro-image-preview (Imagen 4 Preview)
   • Customizable generation parameters (aspect_ratio)
   • Multiple aspect ratio options for different use cases

2. Output Management:
   • Automatic local saving with intelligent filename generation
   • Duplicate filename detection and resolution
   • Organized output directory structure

3. Response Format:
   • Rich response with both text and image data
   • Status tracking and error handling
   • Direct image data for immediate display
   • File path reference for local access

Environment Variables:
    GOOGLE_API_KEY: Your Google Gemini API key (required)
    GEMINI_MODEL_ID: Model to use (optional, defaults to gemini-3-pro-image-preview)

Parameters:
    prompt (str): The text prompt for image generation (required)
    model_id (str): Model identifier - one of the supported Gemini models
    aspect_ratio (str): Aspect ratio for generated images (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)


Usage with Strands Agent:
```python
import os
from strands import Agent
from strands_tools import generate_image_gemini

# Set your API key as environment variable
os.environ['GOOGLE_API_KEY'] = 'your-api-key-here'

# Create agent with the tool
agent = Agent(tools=[generate_image_gemini])

# Basic usage with default parameters
agent.tool.generate_image_gemini(prompt="A robot holding a red skateboard")

# Advanced usage with custom parameters
agent.tool.generate_image_gemini(
    prompt="A futuristic city with flying cars at sunset",
    model_id="gemini-2.5-flash-image",
    aspect_ratio="16:9"
)

# Using different aspect ratio
agent.tool.generate_image_gemini(
    prompt="A serene mountain landscape",
    aspect_ratio="4:3"
)
```

For more information about Google Gemini image generation, see:
https://ai.google.dev/gemini-api/docs/image-generation

See the generate_image_gemini function docstring for more details on parameters and options.
"""

import datetime
import hashlib
import logging
import os
import re
import uuid
from typing import Any

from strands.types.tools import ToolResult, ToolUse

# Set up logger for this module
logger = logging.getLogger(__name__)

# Constants
MAX_FILENAME_LENGTH = 100
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_MODEL_ID = "gemini-3-pro-image-preview"

# Valid parameter values for validation
VALID_MODEL_IDS = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]

# Aspect ratio to resolution mapping for Gemini image generation
ASPECT_RATIO_TO_RESOLUTION = {
    "1:1": {"width": 1024, "height": 1024},
    "2:3": {"width": 832, "height": 1248},
    "3:2": {"width": 1248, "height": 832},
    "3:4": {"width": 864, "height": 1184},
    "4:3": {"width": 1184, "height": 864},
    "4:5": {"width": 896, "height": 1152},
    "5:4": {"width": 1152, "height": 896},
    "9:16": {"width": 768, "height": 1344},
    "16:9": {"width": 1344, "height": 768},
    "21:9": {"width": 1536, "height": 672},
}

VALID_ASPECT_RATIOS = list(ASPECT_RATIO_TO_RESOLUTION.keys())

TOOL_SPEC = {
    "name": "generate_image_gemini",
    "description": "Generates images using Google's Gemini models based on text prompts",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt for image generation",
                },
                "model_id": {"type": "string", "description": "Model ID for image generation."},
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio for generated images",
                    "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                },
            },
            "required": ["prompt"],
        }
    },
}


def create_filename(prompt: str) -> str:
    """
    Generate a filename from the prompt text.

    Extracts the first 5 words from the prompt, sanitizes special characters,
    and limits the filename length to MAX_FILENAME_LENGTH characters.

    Args:
        prompt: The text prompt used for image generation.

    Returns:
        A sanitized filename string derived from the prompt.
    """
    # Extract first 5 words (alphanumeric sequences)
    words = re.findall(r"\w+", prompt.lower())[:5]
    # Join words with underscores
    filename = "_".join(words)
    # Sanitize: remove any remaining special characters except underscores, hyphens, and dots
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    # Limit filename length
    return filename[:MAX_FILENAME_LENGTH]


def _sanitize_error_message(error_msg: str, api_key: str | None = None) -> str:
    """
    Sanitize error messages to ensure sensitive information is not exposed.

    This function removes API keys and other sensitive data from error messages
    before they are returned to the user.

    Args:
        error_msg: The original error message.
        api_key: The API key to redact (if any).

    Returns:
        A sanitized error message with sensitive data redacted.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    if api_key and api_key in error_msg:
        error_msg = error_msg.replace(api_key, "[REDACTED]")

    # Also redact any partial API key matches (in case of truncation)
    if api_key and len(api_key) > 8:
        # Redact if at least 8 consecutive characters of the key appear
        for i in range(len(api_key) - 7):
            partial_key = api_key[i : i + 8]
            if partial_key in error_msg:
                error_msg = error_msg.replace(partial_key, "[REDACTED]")

    return error_msg


def _create_error_result(tool_use_id: str, error_msg: str, api_key: str | None = None) -> ToolResult:
    """
    Create a standardized error ToolResult with sanitized message.

    Args:
        tool_use_id: The tool use identifier.
        error_msg: The error message to include.
        api_key: The API key to redact from the message.

    Returns:
        A ToolResult dictionary with error status.
    """
    sanitized_msg = _sanitize_error_message(error_msg, api_key)
    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": f"Error generating image: {sanitized_msg}"}],
    }


def _validate_parameters(tool_input: dict) -> tuple[str, str, str | None]:
    """
    Validate and extract parameters from tool input.

    Args:
        tool_input: Dictionary containing the tool input parameters.

    Returns:
        Tuple of (prompt, model_id, aspect_ratio).

    Raises:
        ValueError: If any parameter is invalid.
    """
    # Validate prompt
    prompt = tool_input.get("prompt", "")
    if not prompt:
        raise ValueError("Prompt is required for image generation.")
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string.")

    # Get and validate model_id
    model_id = tool_input.get("model_id", os.environ.get("GEMINI_MODEL_ID", DEFAULT_MODEL_ID))
    if model_id not in VALID_MODEL_IDS:
        raise ValueError(f"Invalid model_id '{model_id}'. Supported values are: {', '.join(VALID_MODEL_IDS)}")

    # Get and validate aspect_ratio (optional)
    aspect_ratio = tool_input.get("aspect_ratio")
    if aspect_ratio is not None and aspect_ratio not in VALID_ASPECT_RATIOS:
        valid_ratios = ", ".join(VALID_ASPECT_RATIOS)
        raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'. Supported values are: {valid_ratios}")

    return prompt, model_id, aspect_ratio


def call_gemini_api(
    prompt: str,
    model_id: str,
    api_key: str,
    aspect_ratio: str | None = None,
) -> tuple[bytes, str]:
    """
    Generate images using Google Gemini API.

    Args:
        prompt: Text prompt for image generation.
        model_id: Gemini model identifier.
        api_key: Google API key.
        aspect_ratio: Optional aspect ratio.

    Returns:
        Tuple of (image_bytes, finish_reason).

    Raises:
        ImportError: If google-genai package not installed.
        Exception: For API errors.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        response_modalities=["Image"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
        ),
    )

    response = client.models.generate_content(
        model=model_id,
        contents=[prompt],
        config=config,
    )

    # Extract image bytes from response parts
    for part in response.parts:
        if part.inline_data is not None:
            return part.inline_data.data, "SUCCESS"

    raise ValueError("No image data in API response")


def generate_image_gemini(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Generate images from text prompts using Google Gemini models.

    This function transforms textual descriptions into high-quality images using
    Google's Gemini models with native image generation capability. It handles
    the complete process from API interaction to image storage and result formatting.

    How It Works:
    ------------
    1. Extracts and validates parameters from the tool input
    2. Retrieves API key from GOOGLE_API_KEY environment variable
    3. Configures the request with appropriate parameters
    4. Invokes the Google Gemini API for image generation using generate_content
    5. Processes the response to extract image data from inline_data
    6. Creates appropriate filenames based on the prompt content
    7. Saves images to a local output directory
    8. Returns a success response with both text description and rendered images

    Generation Parameters:
    --------------------
    - prompt: The textual description of the desired image (required)
    - model_id: Specific Gemini model to use (gemini-2.5-flash-image or gemini-3-pro-image-preview, defaults to gemini-3-pro-image-preview)
    - aspect_ratio: Controls the aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)

    Common Usage Scenarios:
    ---------------------
    - Creating illustrations for documents or presentations
    - Generating visual concepts for design projects
    - Visualizing scenes or characters for creative writing
    - Producing custom artwork based on specific descriptions
    - Testing visual ideas before commissioning real artwork

    Args:
        tool: ToolUse object containing the parameters for image generation.
            - toolUseId: Unique identifier for this tool invocation
            - input: Dictionary with generation parameters
        **kwargs: Additional keyword arguments (unused).

    Returns:
        ToolResult: A dictionary containing the result status and content:
            - On success: Contains a text message with saved image paths and
              the rendered images in the content array.
            - On failure: Contains an error message describing what went wrong.

    Notes:
        - Requires GOOGLE_API_KEY environment variable to be set
        - Image files are saved to an "output" directory in the current working directory
        - Filenames are generated based on the first few words of the prompt
        - Duplicate filenames are handled by appending an incrementing number
    """
    tool_use_id = tool.get("toolUseId", "default_id")
    api_key = None

    try:
        tool_input = tool.get("input", {})

        # Retrieve API key from environment
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not set")
            return _create_error_result(
                tool_use_id,
                "GOOGLE_API_KEY environment variable not set. Please set it with your Google Gemini API key.",
            )

        # Validate parameters
        try:
            prompt, model_id, aspect_ratio = _validate_parameters(tool_input)
        except ValueError as e:
            logger.error(f"Parameter validation error: {e}")
            return _create_error_result(tool_use_id, str(e), api_key)

        # Generate image using the API
        try:
            image_bytes, finish_reason = call_gemini_api(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                aspect_ratio=aspect_ratio,
            )
        except ImportError as e:
            logger.error(f"Failed to import google-genai: {e}")
            return _create_error_result(
                tool_use_id,
                "google-genai package is not installed. Install it with: pip install google-genai",
                api_key,
            )
        except Exception as e:
            logger.error(f"API request failed: {e}")
            error_msg = str(e).lower()

            # Handle authentication errors
            if (
                "auth" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or ("invalid" in error_msg and "key" in error_msg)
            ):
                return _create_error_result(
                    tool_use_id,
                    "API authentication failed. Please verify your GOOGLE_API_KEY is valid.",
                    api_key,
                )

            # Handle rate limiting errors
            if "rate" in error_msg or "429" in error_msg or "quota" in error_msg or "limit" in error_msg:
                return _create_error_result(
                    tool_use_id,
                    "API rate limit exceeded. Please wait before making more requests or check your quota.",
                    api_key,
                )

            # Handle content policy violations
            if (
                "policy" in error_msg
                or "safety" in error_msg
                or "blocked" in error_msg
                or ("content" in error_msg and "filter" in error_msg)
            ):
                return _create_error_result(
                    tool_use_id,
                    "Content policy violation. The prompt may contain content that violates Google's usage policies.",
                    api_key,
                )

            # Handle network errors
            if "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                return _create_error_result(
                    tool_use_id,
                    f"Network error occurred while connecting to the API: {_sanitize_error_message(str(e), api_key)}",
                    api_key,
                )

            # Generic API error
            return _create_error_result(
                tool_use_id,
                f"API request failed: {_sanitize_error_message(str(e), api_key)}",
                api_key,
            )

        # Create output directory
        output_dir = DEFAULT_OUTPUT_DIR
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except OSError as e:
            logger.error(f"Failed to create output directory: {e}")
            return _create_error_result(
                tool_use_id,
                f"Failed to create output directory '{output_dir}': {e}",
                api_key,
            )

        # Generate unique filename using timestamp and UUID
        base_filename = create_filename(prompt)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        unique_id = str(uuid.uuid4())[:6]
        filename = f"{base_filename}_{timestamp}_{prompt_hash}_{unique_id}.{DEFAULT_IMAGE_FORMAT}"
        image_path = os.path.join(output_dir, filename)

        # Save image
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except OSError as e:
            logger.error(f"Failed to save image to {image_path}: {e}")
            return _create_error_result(
                tool_use_id,
                f"Failed to save image to '{image_path}': {e}",
                api_key,
            )

        # Get resolution info from aspect ratio mapping
        resolution = ASPECT_RATIO_TO_RESOLUTION.get(aspect_ratio, {})

        # Build response content
        text_msg = f"The generated image has been saved locally to {image_path}."
        content = [
            {"text": text_msg},
            {
                "image": {
                    "format": DEFAULT_IMAGE_FORMAT,
                    "source": {"bytes": image_bytes},
                }
            },
        ]

        logger.info(
            "Successfully generated image",
            extra={
                "model": model_id,
                "image_path": image_path,
                "width": resolution.get("width"),
                "height": resolution.get("height"),
            },
        )

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": content,
        }

    except Exception as e:
        # Catch-all exception handler
        logger.exception(f"Unexpected error in generate_image_gemini: {e}")
        return _create_error_result(tool_use_id, str(e), api_key)
