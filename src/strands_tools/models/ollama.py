"""Create instance of SDK's Ollama model provider."""

from typing import Any

from strands.models.ollama import OllamaModel
from strands.types.models import Model


def instance(**model_config: Any) -> Model:
    """Create instance of SDK's Ollama model provider.

    Args:
        **model_config: Configuration options for the Ollama model.

    Returns:
        Ollama model provider.
    """
    return OllamaModel(**model_config)
