"""
TwelveLabs video chat tool for Strands Agent.

This module provides video understanding and Q&A functionality using TwelveLabs' Pegasus model,
enabling natural language conversations about video content. It supports both direct video IDs
and uploading video files for analysis.

Key Features:
1. Video Analysis:
   • Natural language Q&A about video content
   • Multi-modal understanding (visual and audio)
   • Support for various video formats
   • Automatic video indexing

2. Input Options:
   • Use existing video_id from indexed videos
   • Upload video from file path
   • Configurable temperature for response generation
   • Choice of visual/audio analysis modes

3. Response Format:
   • Natural language answers
   • Context-aware responses
   • Detailed video understanding

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import chat_video

agent = Agent(tools=[chat_video])

# Chat with existing video
result = agent.tool.chat_video(
    prompt="What are the main topics discussed?",
    video_id="existing-video-id"
)

# Chat with new video file
result = agent.tool.chat_video(
    prompt="Describe what happens in this video",
    video_path="/path/to/video.mp4",
    index_id="your-index-id"
)
```

See the chat_video function docstring for more details on available parameters.
"""

import hashlib
import os
from typing import Any, Dict

from strands.types.tools import ToolResult, ToolUse
from twelvelabs import TwelveLabs

TOOL_SPEC = {
    "name": "chat_video",
    "description": """Chat with video content using TwelveLabs' Pegasus model for video understanding.

Key Features:
1. Video Analysis:
   - Natural language Q&A about video content
   - Multi-modal understanding (visual and audio)
   - Support for various video formats
   - Automatic video indexing when needed

2. Input Options:
   - Use existing video_id for indexed videos
   - Upload new video from file path
   - Configurable response generation
   - Choice of analysis modes

3. Response Types:
   - Detailed descriptions
   - Question answering
   - Content summarization
   - Action identification
   - Audio transcription

Usage Examples:
1. Chat with existing video:
   chat_video(prompt="What are the key points?", video_id="video_123")

2. Upload and chat with new video:
   chat_video(
       prompt="Describe the main events",
       video_path="/path/to/video.mp4",
       index_id="your-index-id"
   )

3. Focused analysis:
   chat_video(
       prompt="What is being said in the video?",
       video_id="video_123",
       engine_options=["audio"]
   )

4. Creative responses:
   chat_video(
       prompt="Write a story based on this video",
       video_id="video_123",
       temperature=0.9
   )

Note: Either video_id OR video_path must be provided, not both.""",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language question or instruction about the video",
                },
                "video_id": {
                    "type": "string",
                    "description": "ID of an already indexed video in TwelveLabs",
                },
                "video_path": {
                    "type": "string",
                    "description": "Path to a video file to upload and analyze",
                },
                "index_id": {
                    "type": "string",
                    "description": (
                        "TwelveLabs index ID (required for video uploads). "
                        "Uses TWELVELABS_PEGASUS_INDEX_ID env var if not provided"
                    ),
                },
                "temperature": {
                    "type": "number",
                    "description": "Controls randomness in responses (0.0-1.0). Default: 0.7",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "engine_options": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["visual", "audio"],
                    },
                    "description": "Analysis modes to use. Default: ['visual', 'audio']",
                },
            },
            "required": ["prompt"],
            "oneOf": [
                {"required": ["video_id"]},
                {"required": ["video_path"]},
            ],
        }
    },
}

# Cache for uploaded videos to avoid re-uploading
VIDEO_CACHE: Dict[str, str] = {}


def get_video_hash(video_path: str) -> str:
    """
    Calculate SHA256 hash of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(video_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def upload_and_index_video(video_path: str, index_id: str, api_key: str) -> str:
    """
    Upload a video file to TwelveLabs and wait for indexing.

    Args:
        video_path: Path to the video file
        index_id: TwelveLabs index ID
        api_key: TwelveLabs API key

    Returns:
        video_id of the uploaded video

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video indexing fails
    """
    # Check if file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check cache first
    video_hash = get_video_hash(video_path)
    if video_hash in VIDEO_CACHE:
        return VIDEO_CACHE[video_hash]

    # Upload video
    with TwelveLabs(api_key) as client:
        # Read video file
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Create upload task
        task = client.task.create(index_id=index_id, file=video_bytes)

        # Wait for indexing to complete
        task.wait_for_done(sleep_interval=5)

        if task.status != "ready":
            raise RuntimeError(f"Video indexing failed with status: {task.status}")

        video_id = str(task.video_id)
        VIDEO_CACHE[video_hash] = video_id

        return video_id


def chat_video(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Chat with video content using TwelveLabs Pegasus model.

    This tool enables natural language conversations about video content using
    TwelveLabs' Pegasus model. It can analyze both visual and audio aspects
    of videos to answer questions, provide descriptions, and extract insights.

    How It Works:
    ------------
    1. Takes either an existing video_id or uploads a new video from video_path
    2. Sends your prompt to TwelveLabs' Pegasus model
    3. The model analyzes the video content (visual and/or audio)
    4. Returns a natural language response based on the video understanding

    Common Usage Scenarios:
    ---------------------
    - Summarizing video content
    - Answering specific questions about videos
    - Describing actions and events in videos
    - Extracting dialogue or audio information
    - Identifying objects, people, or scenes
    - Creating video transcripts or captions

    Args:
        tool: Tool use information containing input parameters:
            prompt: Natural language question or instruction
            video_id: ID of existing indexed video (optional)
            video_path: Path to video file to upload (optional)
            index_id: Index ID for uploads (default: from TWELVELABS_PEGASUS_INDEX_ID env)
            temperature: Response randomness 0.0-1.0 (default: 0.7)
            engine_options: Analysis modes ['visual', 'audio'] (default: both)

    Returns:
        Dictionary containing status and Pegasus response:
        {
            "toolUseId": "unique_id",
            "status": "success|error",
            "content": [{"text": "Pegasus response or error message"}]
        }

        Success: Returns the model's natural language response
        Error: Returns information about what went wrong

    Notes:
        - Requires TWELVELABS_API_KEY environment variable
        - For video uploads, index_id is required (or set via TWELVELABS_PEGASUS_INDEX_ID env)
        - Uploaded videos are cached to avoid re-uploading the same file
        - Visual mode analyzes what's seen in the video
        - Audio mode analyzes speech and sounds
        - Using both modes provides the most comprehensive understanding
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    try:
        # Get API key
        api_key = os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TWELVELABS_API_KEY environment variable not set. Please set it to your TwelveLabs API key."
            )

        # Extract parameters
        prompt = tool_input["prompt"]
        video_id = tool_input.get("video_id")
        video_path = tool_input.get("video_path")
        temperature = tool_input.get("temperature", 0.7)
        engine_options = tool_input.get("engine_options", ["visual", "audio"])

        # Validate input - must have either video_id or video_path
        if not video_id and not video_path:
            raise ValueError("Either video_id or video_path must be provided")

        if video_id and video_path:
            raise ValueError("Cannot provide both video_id and video_path. Choose one.")

        # Handle video upload if video_path is provided
        if video_path:
            index_id = tool_input.get("index_id") or os.getenv("TWELVELABS_PEGASUS_INDEX_ID")
            if not index_id:
                raise ValueError(
                    "index_id is required for video uploads. "
                    "Provide it in the request or set TWELVELABS_PEGASUS_INDEX_ID environment variable."
                )

            # Upload and index the video
            video_id = upload_and_index_video(video_path, index_id, api_key)
            upload_note = f"Video uploaded successfully. Video ID: {video_id}\n\n"
        else:
            upload_note = ""

        # Generate response using Pegasus
        with TwelveLabs(api_key) as client:
            response = client.analyze(
                video_id=video_id,
                prompt=prompt,
                temperature=temperature,
            )

            # Extract response text
            if hasattr(response, "data"):
                response_text = str(response.data)
            else:
                response_text = str(response)

            # Build complete response
            full_response = upload_note + response_text

            # Add metadata about the analysis
            metadata_parts = [
                "\n\n---",
                f"Video ID: {video_id}",
                f"Temperature: {temperature}",
                f"Engine options: {', '.join(engine_options)}",
            ]

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": full_response + "\n".join(metadata_parts)}],
            }

    except FileNotFoundError as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"File error: {e!s}"}],
        }

    except Exception as e:
        error_message = f"Error chatting with video: {e!s}"

        # Add helpful context for common errors
        if "api_key" in str(e).lower():
            error_message += "\n\nMake sure TWELVELABS_API_KEY environment variable is set correctly."
        elif "index" in str(e).lower():
            error_message += (
                "\n\nMake sure the index_id is valid and you have access to it. "
                "For video uploads, index_id is required."
            )
        elif "video" in str(e).lower() and "not found" in str(e).lower():
            error_message += "\n\nThe specified video_id was not found. Make sure it exists in your index."
        elif "throttl" in str(e).lower() or "rate" in str(e).lower():
            error_message += "\n\nAPI rate limit exceeded. Please try again later."
        elif "task" in str(e).lower() or "upload" in str(e).lower():
            error_message += (
                "\n\nVideo upload or processing failed. "
                "Check that the video file is valid and the index supports video uploads."
            )

        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_message}],
        }
