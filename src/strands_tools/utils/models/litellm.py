"""Create instance of SDK's LiteLLM model provider."""

from strands.models import Model
from strands.models.litellm import LiteLLMModel
from typing_extensions import Unpack


def instance(**model_config: Unpack[LiteLLMModel.LiteLLMConfig]) -> Model:
    """Create instance of SDK's LiteLLM model provider.
    Args:
        **model_config: Configuration options for the LiteLLM model.
    Returns:
        LiteLLM model provider.
    """
    return LiteLLMModel(**model_config)
