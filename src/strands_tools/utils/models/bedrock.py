"""Create instance of SDK's Bedrock model provider."""

from botocore.config import Config as BotocoreConfig
from strands.models import BedrockModel, Model
from typing_extensions import Unpack


def instance(**model_config: Unpack[BedrockModel.BedrockConfig]) -> Model:
    """Create instance of SDK's Bedrock model provider.
    Args:
        **model_config: Configuration options for the Bedrock model.
    Returns:
        Bedrock model provider.
    """
    # Handle conversion of boto_client_config from dict to BotocoreConfig
    if "boto_client_config" in model_config and isinstance(model_config["boto_client_config"], dict):
        model_config["boto_client_config"] = BotocoreConfig(**model_config["boto_client_config"])

    return BedrockModel(**model_config)
