import os
import re
import time
import uuid
from unittest.mock import patch

import boto3
import pytest
from strands import Agent
from strands_tools import nova_reels

AWS_REGION = "us-east-1"
TEST_BUCKET = f"nova-reels-e2e-test-{str(uuid.uuid4())[:8]}".lower()
TIMEOUT_SECONDS = 600
POLL_INTERVAL = 15

pytestmark = pytest.mark.skip(reason="Integration tests are flaky, disabling until they can be made reliable.")

@pytest.fixture
def s3_bucket():
    """Create a temporary S3 bucket and remove it after the test."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.create_bucket(Bucket=TEST_BUCKET)
    yield TEST_BUCKET
    # Clean up
    response = s3.list_objects_v2(Bucket=TEST_BUCKET)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3.delete_object(Bucket=TEST_BUCKET, Key=obj["Key"])
    s3.delete_bucket(Bucket=TEST_BUCKET)


@pytest.fixture
def agent():
    return Agent(tools=[nova_reels])


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_generate_video_store_to_s3(agent, s3_bucket):
    """Test nova reels."""
    prompt = (
        f"Generate a 1-second video showing a robot waving its hand. "
        f"Store the result in S3 bucket {s3_bucket}. Start the video generation job now."
    )

    reply = agent(prompt)
    reply_text = str(reply).lower()

    # Extract the Bedrock async invoke ARN from agent's reply to track job status
    arn_match = re.search(r"arn:aws:bedrock:[a-z0-9-]+:[0-9]{12}:async-invoke/[a-z0-9]{12}", reply_text)
    assert arn_match, f"No valid Bedrock async invoke ARN found in agent reply: {reply_text}"
    arn = arn_match.group(0)

    # Use the nova_reels tool via the agent to check job status
    status = "InProgress"
    current_time = time.time()
    while status == "InProgress" and (time.time() - current_time) < TIMEOUT_SECONDS:
        status_response = agent.tool.nova_reels(action="status", invocation_arn=arn)
        status_content = status_response.get("content", [])
        status_texts = [item["text"].lower() for item in status_content]

        if any("completed" in text for text in status_texts):
            status = "Completed"
            break
        elif any("failed" in text for text in status_texts):
            failure_msg = next((text for text in status_texts if "error:" in text), "Unknown failure")
            pytest.fail(f"Nova Reels job failed: {failure_msg}")

        time.sleep(POLL_INTERVAL)
    else:
        pytest.fail("Nova Reels video generation timed out after 10 minutes")

    # Check S3 bucket is not empty
    s3 = boto3.client("s3", region_name=AWS_REGION)
    all_objects = s3.list_objects_v2(Bucket=s3_bucket)
    if "Contents" not in all_objects:
        pytest.fail(f"No objects found in bucket {s3_bucket}")

    # Find MP4 files in the bucket
    mp4_files = [obj for obj in all_objects["Contents"] if obj["Key"].endswith(".mp4")]
    assert len(mp4_files) > 0, f"No MP4 files found in bucket {s3_bucket}"

    # We should only have 1 MP4 file
    mp4_file = mp4_files[0]
    file_key = mp4_file["Key"]
    file_size = mp4_file["Size"]

    # Validate file size
    assert file_size > 0, f"MP4 file is empty: {file_size} bytes"
    assert file_size > 1000, f"MP4 file too small (likely corrupted): {file_size} bytes"

    # Verify it's actually an MP4 file by checking first few bytes
    try:
        obj_data = s3.get_object(Bucket=s3_bucket, Key=file_key, Range="bytes=0-11")
        first_bytes = obj_data["Body"].read()

        is_mp4 = b"ftyp" in first_bytes[:12] or first_bytes.startswith(b"\x00\x00\x00") and b"ftyp" in first_bytes
        assert is_mp4, f"File doesn't appear to be a valid MP4: {first_bytes.hex()}"
    except Exception as e:
        pytest.fail(f"Error validating MP4 file content: {e}")
