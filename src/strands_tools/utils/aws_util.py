"""AWS utility functions for region resolution."""

import os
from typing import Optional

import boto3
from strands.models.bedrock import DEFAULT_BEDROCK_REGION


def resolve_region(region_name: Optional[str] = None) -> str:
    """Resolve AWS region with fallback hierarchy."""
    if region_name:
        return region_name

    try:
        session = boto3.Session()
        if session.region_name:
            return session.region_name
    except Exception:
        pass

    env_region = os.environ.get("AWS_REGION")
    if env_region:
        return env_region

    return DEFAULT_BEDROCK_REGION
