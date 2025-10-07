"""
Image processing tool for Strands Agent.

This module provides functionality to read image files from disk and convert them
into the required format for use with the Converse API in Strands Agent. It supports
various image formats including PNG, JPEG, GIF, and WebP, with automatic format detection.

Key Features:
1. Image Processing:
   • Automatic format detection (PNG, JPEG/JPG, GIF, WebP)
   • Binary file handling
   • Error handling for invalid files

2. File Path Handling:
   • Support for absolute paths
   • User directory expansion (~/path/to/image.png)
   • Path validation and error reporting

3. Response Format:
   • Properly formatted image content for Converse API
   • Format-specific processing
   • Binary data conversion

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import image_reader

agent = Agent(tools=[image_reader])

# Basic usage - read an image file
result = agent.tool.image_reader(image_path="/path/to/image.jpg")

# With user directory path
result = agent.tool.image_reader(image_path="~/Documents/images/photo.png")
```

See the image_reader function docstring for more details on parameters and return format.
"""

import os
from os.path import expanduser
from typing import Any

from PIL import Image
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "image_reader",
    "description": "Reads an image file from a given path and returns it in the format required for the Converse API",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the image file",
                }
            },
            "required": ["image_path"],
        }
    },
}


def image_reader(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Read an image file from disk and prepare it for use with Converse API.

    This function reads image files from the specified path, detects the image format,
    and converts the content into the proper format required by the Converse API.
    It handles various image formats and provides appropriate error messages when
    issues are encountered.

    How It Works:
    ------------
    1. The function expands the provided path (handling ~/ notation)
    2. It checks if the file exists at the specified path
    3. The image file is read as binary data
    4. PIL/Pillow is used to detect the image format
    5. The image data is formatted for the Converse API with proper format identification

    Common Usage Scenarios:
    ---------------------
    - Visual analysis: Loading images for AI-based analysis
    - Document processing: Loading scanned documents for text extraction
    - Multimodal inputs: Combining image and text inputs for comprehensive tasks
    - Image verification: Loading images to verify their validity or contents

    Args:
        tool: ToolUse object containing the tool usage information and parameters
              The tool input should include:
              - image_path (str): Path to the image file to read. Can be absolute
                or user-relative (with ~/).
        **kwargs: Additional keyword arguments (not used in this function)

    Returns:
        ToolResult: A dictionary containing the status and content:
        - On success: Returns image data formatted for the Converse API
          {
              "toolUseId": "<tool_use_id>",
              "status": "success",
              "content": [{"image": {"format": "<image_format>", "source": {"bytes": <binary_data>}}}]
          }
        - On failure: Returns an error message
          {
              "toolUseId": "<tool_use_id>",
              "status": "error",
              "content": [{"text": "Error message"}]
          }

    Notes:
        - Supported image formats include: PNG, JPEG/JPG, GIF, and WebP
        - If the image format is not recognized, it defaults to PNG
        - The function validates file existence before attempting to read
        - User paths with tilde (~) are automatically expanded
    """
    try:
        tool_use_id = tool["toolUseId"]
        tool_input = tool["input"]

        if "image_path" not in tool_input:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "File path is required"}],
            }

        file_path = expanduser(tool_input.get("image_path"))

        if not os.path.exists(file_path):
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"File not found at path: {file_path}"}],
            }

        with open(file_path, "rb") as file:
            file_bytes = file.read()

        # Handle image files using PIL
        with Image.open(file_path) as img:
            image_format = img.format.lower()
            if image_format not in ["png", "jpeg", "jpg", "gif", "webp"]:
                image_format = "png"  # Default to PNG if format is not recognized

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"image": {"format": image_format, "source": {"bytes": file_bytes}}}],
        }
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error reading file: {str(e)}"}],
        }
