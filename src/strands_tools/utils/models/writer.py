"""Create instance of SDK's OpenAI model provider."""

from typing import Any

from strands.models import Model
from strands.models.writer import WriterModel


def instance(**model_config: Any) -> Model:
    """Create instance of SDK's Writer model provider.
    Args:
        **model_config: Configuration options for the Writer model.
    Returns:
        Writer model provider.
    """
    return WriterModel(**model_config)
