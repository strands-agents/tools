"""Create instance of SDK's LlamaAPI model provider."""

from strands.models import Model
from strands.models.llamaapi import LlamaAPIModel
from typing_extensions import Unpack


def instance(**model_config: Unpack[LlamaAPIModel.LlamaConfig]) -> Model:
    """Create instance of SDK's LlamaAPI model provider.
    Args:
        **model_config: Configuration options for the LlamaAPI model.
    Returns:
        LlamaAPI model provider.
    """
    return LlamaAPIModel(**model_config)
