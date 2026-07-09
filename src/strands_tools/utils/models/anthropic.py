"""Create instance of SDK's Anthropic model provider."""

from typing import Any

from strands.models import Model
from strands.models.anthropic import AnthropicModel


def instance(**model_config: Any) -> Model:
    """Create instance of SDK's Anthropic model provider.
    Args:
        **model_config: Configuration options for the Anthropic model.
    Returns:
        Anthropic model provider.
    """
    return AnthropicModel(**model_config)
