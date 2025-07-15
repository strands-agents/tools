"""Create instance of SDK's Ollama model provider."""

from typing import Any

from strands.models import Model
from strands.models.ollama import OllamaModel


def instance(**model_config: Any) -> Model:
    """Create instance of SDK's Ollama model provider.
    Args:
        **model_config: Configuration options for the Ollama model.
    Returns:
        Ollama model provider.
    """
    return OllamaModel(**model_config)
