"""Create instance of SDK's OpenAI model provider."""

from typing import Any

from strands.models import Model
from strands.models.openai import OpenAIModel


def instance(**model_config: Any) -> Model:
    """Create instance of SDK's OpenAI model provider.
    Args:
        **model_config: Configuration options for the OpenAI model.
    Returns:
        OpenAI model provider.
    """
    return OpenAIModel(**model_config)
