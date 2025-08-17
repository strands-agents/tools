"""
Nova Reels video generation tool for Amazon Bedrock.

This module provides functionality to create high-quality videos using Amazon Bedrock's
Nova Reel model. It supports both text-to-video (T2V) and image-to-video (I2V) generation
with configurable parameters.

Key Features:
1. Text-to-Video Generation:
   - Create videos from text descriptions
   - Configure video quality and resolution
   - Set custom seeds for deterministic results

2. Image-to-Video Generation:
   - Transform static images into dynamic videos
   - Apply text prompts to guide animation style
   - Support for common image formats

3. Job Management:
   - Create new video generation jobs
   - Check job status and progress
   - List and filter existing jobs

4. Output Control:
   - Direct output to specified S3 buckets
   - Standard video format (MP4)
   - Configurable resolution and FPS

Usage Examples:
```python
# Text to Video generation
agent.tool.nova_reels(
    action="create",
    text="A cinematic shot of a giraffe walking through a savanna at sunset",
    s3_bucket="my-video-output-bucket"
)

# Image to Video generation with custom parameters
agent.tool.nova_reels(
    action="create",
    text="Transform this forest scene into autumn with falling leaves",
    image_path="/path/to/forest_image.jpg",
    s3_bucket="my-video-output-bucket",
    seed=42,
    fps=30,
    dimension="1920x1080"
)

# Check video generation status
agent.tool.nova_reels(
    action="status",
    invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-inference/..."
)

# List video generation jobs with custom region
# First set environment variable: export AWS_REGION=us-east-1
agent.tool.nova_reels(
    action="list",
    max_results=5,
    status_filter="Completed"
)
```

The videos are generated asynchronously, and completion typically takes 5-10 minutes.
Results are stored in the specified S3 bucket and can be accessed once the job completes.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands import tool

from strands_tools.utils import console_util

# Environment variables for configurable default parameters


@tool
def nova_reels(
    action: str,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    seed: int = None,
    fps: int = None,
    dimension: str = None,
    invocation_arn: Optional[str] = None,
    max_results: int = None,
    status_filter: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create high-quality videos using Amazon Nova Reel.

    This tool interfaces with Amazon Bedrock's Nova Reel model to generate professional-quality
    videos from text descriptions or input images. It supports text-to-video (T2V) and
    image-to-video (I2V) generation, as well as job status checking and listing.

    How It Works:
    -------------
    1. For video creation:
       - Configures request parameters based on inputs
       - Connects to Bedrock Runtime API in configured region
       - Submits asynchronous job for video generation
       - Returns job ARN for status tracking

    2. For status checking:
       - Fetches current status of a specific job by ARN
       - Returns completion status, error information, or progress details

    3. For job listing:
       - Retrieves a list of submitted jobs with their status
       - Supports filtering by job status and pagination

    Operation Modes:
    --------------
    1. Create (Text-to-Video):
       - Requires text prompt and S3 bucket
       - Configurable fps, and dimension
       - Optional seed parameter for reproducible results

    2. Create (Image-to-Video):
       - Requires text prompt, image path, and S3 bucket
       - Transforms input image according to text prompt
       - Creates animation from static image with configurable parameters

    3. Status Check:
       - Requires invocation ARN from a previous create operation
       - Returns current job status (Completed, InProgress, Failed)
       - Includes output location when job is complete

    4. Job Listing:
       - Lists recent video generation jobs
       - Can filter by job status
       - Supports limiting results count

    Args:
        action: Action to perform. Must be one of "create", "status", or "list".
        text: Text prompt describing the desired video content. Required for "create" action.
        image_path: Optional path to an image for image-to-video generation.
            If provided along with text, generates a video that transforms the image according to the text prompt.
        s3_bucket: S3 bucket name where the generated video will be stored. Required for "create" action.
        seed: Optional seed integer for video generation. Using the same seed and prompt will
            produce similar results. Default is controlled by NOVA_REEL_DEFAULT_SEED env variable (default: 0).
        fps: Frames per second for the generated video. Default is controlled by NOVA_REEL_DEFAULT_FPS
            env variable (default: 24). Common values are 24, 30, or 60.
        dimension: Video resolution in "WIDTHxHEIGHT" format. Default is controlled by NOVA_REEL_DEFAULT_DIMENSION
            env variable (default: "1280x720"). Common values are "1280x720" (720p) or "1920x1080" (1080p).
        invocation_arn: Required for "status" action. The ARN of the video generation job
            returned from a previous create operation.
        max_results: Optional maximum number of jobs to return when using the "list" action.
            Default is controlled by NOVA_REEL_DEFAULT_MAX_RESULTS env variable (default: 10).
        status_filter: Optional filter for the "list" action to only return jobs with this status.
            Must be one of "Completed", "InProgress", or "Failed".
        region: AWS region to use. If not provided, will use the AWS_REGION environment
            variable, falling back to "us-east-1" if not set.

    Returns:
        Dict containing operation status and results:
        - For "create": Job ARN and submission confirmation
        - For "status": Current job status and output location if complete
        - For "list": List of jobs with their details

        Success format:
        {
            "status": "success",
            "content": [
                {"text": "Operation-specific message"},
                {"text": "Additional details or data"}
            ]
        }

        Error format:
        {
            "status": "error",
            "content": [
                {"text": "Error: [error message]"}
            ]
        }

    Notes:
        - Video generation typically takes 5-10 minutes to complete
        - The Bedrock Nova Reel model is available in specific regions only, default is us-east-1
        - Videos can be configured for fps, and resolution
        - For image-to-video, the input image should ideally match the output video dimensions
        - S3 buckets must be accessible to the AWS credentials used for Bedrock
        - Set AWS_REGION environment variable to change the default region
    """
    console = console_util.create()

    seed = int(os.getenv("NOVA_REEL_DEFAULT_SEED", "0")) if seed is None else seed
    fps = int(os.getenv("NOVA_REEL_DEFAULT_FPS", "24")) if fps is None else fps
    dimension = os.getenv("NOVA_REEL_DEFAULT_DIMENSION", "1280x720") if dimension is None else dimension
    max_results = int(os.getenv("NOVA_REEL_DEFAULT_MAX_RESULTS", "10")) if max_results is None else max_results
    region = os.getenv("AWS_REGION", "us-east-1") if region is None else region
    try:
        console.print("\n🚀 Nova Reels Tool - Starting Execution")
        console.print(f"Action requested: {action}")

        # Get region from parameter, environment variable, or default to us-east-1
        aws_region = region
        console.print(f"📡 Connecting to Bedrock Runtime in {aws_region}")

        # Create Bedrock Runtime client with configurable region
        config = BotocoreConfig(user_agent_extra="strands-agents-nova-reels")
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=aws_region, config=config)

        if action == "create":
            if not text:
                raise ValueError("Text prompt is required for video generation")

            if not s3_bucket:
                raise ValueError("S3 bucket is required for video output")

            # Parse dimensions to ensure proper format
            try:
                width, height = map(int, dimension.split("x"))
                if width <= 0 or height <= 0:
                    raise ValueError("Width and height must be positive integers")
            except Exception:
                raise ValueError("dimension must be in format 'WIDTHxHEIGHT', e.g. '1280x720'") from None

            model_input = {
                "taskType": "TEXT_VIDEO",
                "textToVideoParams": {"text": text},
                "videoGenerationConfig": {
                    "durationSeconds": 6,
                    "fps": fps,
                    "dimension": dimension,
                    "seed": seed,
                },
            }

            # Handle image-to-video if image path provided
            if image_path:
                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                    model_input["textToVideoParams"]["images"] = [
                        {
                            "format": Path(image_path).suffix[1:],
                            "source": {"bytes": image_base64},
                        }
                    ]
                except Exception as e:
                    raise ValueError(f"Failed to process input image: {str(e)}") from e

            # Start async video generation
            console.print("\n📼 Starting video generation:")
            console.print(f"🎯 Target S3 bucket: s3://{s3_bucket}")
            console.print(f"📝 Text prompt: {text}")
            if image_path:
                console.print(f"🖼️ Using input image: {image_path}")
            console.print(
                "⚙️ Model configuration:",
                json.dumps(model_input["videoGenerationConfig"], indent=2),
            )

            invocation = bedrock_runtime.start_async_invoke(
                modelId="amazon.nova-reel-v1:1",
                modelInput=model_input,
                outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_bucket}"}},
            )
            console.print(f"✨ Job started with ARN: {invocation['invocationArn']}")

            return {
                "status": "success",
                "content": [
                    {"text": "Video generation job started successfully"},
                    {"text": f"Task ARN: {invocation['invocationArn']}"},
                    {
                        "text": (
                            "Note: Video generation typically takes 5-10 minutes. Use the 'status' action to check "
                            "progress."
                        )
                    },
                ],
            }

        elif action == "status":
            if not invocation_arn:
                raise ValueError("invocation_arn is required to check status")

            console.print(f"\n🔍 Checking status for job: {invocation_arn}")
            invocation = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)

            status = invocation["status"]
            console.print(f"📊 Current status: {status}")
            messages = []

            if status == "Completed":
                bucket_uri = invocation["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                video_uri = f"{bucket_uri}/output.mp4"
                messages = [
                    {"text": "✅ Video generation completed!"},
                    {"text": f"Video available at: {video_uri}"},
                ]
            elif status == "InProgress":
                start_time = invocation["submitTime"]
                messages = [
                    {"text": "⏳ Job in progress"},
                    {"text": f"Started at: {start_time}"},
                ]
            elif status == "Failed":
                failure_message = invocation.get("failureMessage", "Unknown error")
                messages = [
                    {"text": "❌ Job failed"},
                    {"text": f"Error: {failure_message}"},
                ]

            return {"status": "success", "content": messages}

        elif action == "list":
            console.print(f"\n📋 Listing jobs (max: {max_results})")
            if status_filter:
                console.print(f"🔍 Filtering by status: {status_filter}")

            list_args = {"maxResults": max_results}
            if status_filter:
                list_args["statusEquals"] = status_filter

            jobs = bedrock_runtime.list_async_invokes(**list_args)

            return {
                "status": "success",
                "content": [
                    {"text": f"Found {len(jobs['asyncInvokeSummaries'])} jobs:"},
                    {"text": json.dumps(jobs, indent=2, default=str)},
                ],
            }
        else:
            raise ValueError(f"Unknown action '{action}'. Must be one of: create, status, list")

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
