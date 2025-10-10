"""Utilities for loading model providers in strands_tools."""

import importlib
import json
import os
import pathlib
from typing import Any

from botocore.config import Config
from strands.models import Model

# Default model configuration for Bedrock
DEFAULT_MODEL_CONFIG = {
    "model_id": os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
    "max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "10000")),
    "boto_client_config": Config(
        read_timeout=int(os.getenv("STRANDS_BOTO_READ_TIMEOUT", "900")),
        connect_timeout=int(os.getenv("STRANDS_BOTO_CONNECT_TIMEOUT", "900")),
        retries=dict(
            max_attempts=int(os.getenv("STRANDS_BOTO_MAX_ATTEMPTS", "3")),
            mode="adaptive",
        ),
    ),
    "additional_request_fields": {},
    "cache_tools": os.getenv("STRANDS_CACHE_TOOLS", "default"),
    "cache_prompt": os.getenv("STRANDS_CACHE_PROMPT", "default"),
}

# Parse additional request fields if provided
ADDITIONAL_REQUEST_FIELDS = os.getenv("STRANDS_ADDITIONAL_REQUEST_FIELDS", "{}")
if ADDITIONAL_REQUEST_FIELDS != "{}":
    try:
        DEFAULT_MODEL_CONFIG["additional_request_fields"] = json.loads(ADDITIONAL_REQUEST_FIELDS)
    except json.JSONDecodeError:
        pass

# Add anthropic beta features if specified
ANTHROPIC_BETA_FEATURES = os.getenv("STRANDS_ANTHROPIC_BETA", "")
if len(ANTHROPIC_BETA_FEATURES) > 0:
    DEFAULT_MODEL_CONFIG["additional_request_fields"]["anthropic_beta"] = ANTHROPIC_BETA_FEATURES.split(",")

# Add thinking configuration if specified
THINKING_TYPE = os.getenv("STRANDS_THINKING_TYPE", "")
BUDGET_TOKENS = os.getenv("STRANDS_BUDGET_TOKENS", "")
if THINKING_TYPE:
    thinking_config = {"type": THINKING_TYPE}
    if BUDGET_TOKENS:
        thinking_config["budget_tokens"] = int(BUDGET_TOKENS)
    DEFAULT_MODEL_CONFIG["additional_request_fields"]["thinking"] = thinking_config


def load_path(name: str) -> pathlib.Path:
    """Locate the model provider module file path.

    First search "$CWD/.models". If the module file is not found, fall back to the built-in models directory.

    Args:
        name: Name of the model provider (e.g., bedrock).

    Returns:
        The file path to the model provider module.

    Raises:
        ImportError: If the model provider module cannot be found.
    """
    path = pathlib.Path.cwd() / ".models" / f"{name}.py"
    if not path.exists():
        path = pathlib.Path(__file__).parent / ".." / "models" / f"{name}.py"

    if not path.exists():
        raise ImportError(f"model_provider=<{name}> | does not exist")

    return path


def load_config(config: str) -> dict[str, Any]:
    """Load model configuration from a JSON string or file.

    Args:
        config: A JSON string or path to a JSON file containing model configuration.
            If empty string or '{}', the default config is used.

    Returns:
        The parsed configuration.
    """
    if not config or config == "{}":
        return DEFAULT_MODEL_CONFIG

    if config.endswith(".json"):
        with open(config) as fp:
            return json.load(fp)

    return json.loads(config)


def load_model(path: pathlib.Path, config: dict[str, Any]) -> Model:
    """Dynamically load and instantiate a model provider from a Python module.

    Imports the module at the specified path and calls its 'instance' function
    with the provided configuration to create a model instance.

    Args:
        path: Path to the Python module containing the model provider implementation.
        config: Configuration to pass to the model provider's instance function.

    Returns:
        An instantiated model provider.
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.instance(**config)


def create_model(provider: str = None, config: dict[str, Any] = None) -> Model:
    """Create model based on provider configuration.

    Args:
        provider: Model provider name. If None, uses STRANDS_PROVIDER env var or defaults to 'bedrock'.
        config: Model configuration dict. If None, uses environment-based config.

    Returns:
        Configured model instance.
    """
    if provider is None:
        provider = os.getenv("STRANDS_PROVIDER", "bedrock")

    if config is None:
        config = get_provider_config(provider)

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        return BedrockModel(**config)

    elif provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(**config)

    elif provider == "litellm":
        from strands.models.litellm import LiteLLMModel

        return LiteLLMModel(**config)

    elif provider == "llamaapi":
        from strands.models.llamaapi import LlamaAPIModel

        return LlamaAPIModel(**config)

    elif provider == "ollama":
        from strands.models.ollama import OllamaModel

        return OllamaModel(**config)

    elif provider == "openai":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(**config)

    elif provider == "writer":
        from strands.models.writer import WriterModel

        return WriterModel(**config)

    elif provider == "cohere":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(**config)

    elif provider == "github":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(**config)

    else:
        # Try to load custom model provider
        try:
            path = load_path(provider)
            return load_model(path, config)
        except ImportError:
            raise ValueError(f"Unknown model provider: {provider}") from None


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific model provider based on environment variables.

    Args:
        provider: Model provider name.

    Returns:
        Configuration dictionary for the provider.
    """
    if provider == "bedrock":
        return {
            "model_id": os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
            "max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "10000")),
            "boto_client_config": Config(
                read_timeout=900,
                connect_timeout=900,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
            "additional_request_fields": DEFAULT_MODEL_CONFIG["additional_request_fields"],
            "cache_prompt": os.getenv("STRANDS_CACHE_PROMPT", "default"),
            "cache_tools": os.getenv("STRANDS_CACHE_TOOLS", "default"),
        }

    elif provider == "anthropic":
        return {
            "client_args": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
            "max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "10000")),
            "model_id": os.getenv("STRANDS_MODEL_ID", "claude-sonnet-4-20250514"),
            "params": {
                "temperature": float(os.getenv("STRANDS_TEMPERATURE", "1")),
            },
        }

    elif provider == "litellm":
        client_args = {"api_key": os.getenv("LITELLM_API_KEY")}
        if os.getenv("LITELLM_BASE_URL"):
            client_args["base_url"] = os.getenv("LITELLM_BASE_URL")

        return {
            "client_args": client_args,
            "model_id": os.getenv("STRANDS_MODEL_ID", "anthropic/claude-sonnet-4-20250514"),
            "params": {
                "max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "10000")),
                "temperature": float(os.getenv("STRANDS_TEMPERATURE", "1")),
            },
        }

    elif provider == "llamaapi":
        return {
            "client_args": {
                "api_key": os.getenv("LLAMAAPI_API_KEY"),
            },
            "model_id": os.getenv("STRANDS_MODEL_ID", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
            "params": {
                "max_completion_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("STRANDS_TEMPERATURE", "1")),
            },
        }

    elif provider == "ollama":
        return {
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "model_id": os.getenv("STRANDS_MODEL_ID", "qwen3:4b"),
        }

    elif provider == "openai":
        return {
            "client_args": {"api_key": os.getenv("OPENAI_API_KEY")},
            "model_id": os.getenv("STRANDS_MODEL_ID", "o4-mini"),
            "params": {"max_completion_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "10000"))},
        }

    elif provider == "writer":
        return {
            "client_args": {
                "api_key": os.getenv("WRITER_API_KEY"),
            },
            "model_id": os.getenv("STRANDS_MODEL_ID", "palmyra-x5"),
        }

    elif provider == "cohere":
        return {
            "client_args": {
                "api_key": os.getenv("COHERE_API_KEY"),
                "base_url": "https://api.cohere.ai/compatibility/v1",
            },
            "model_id": os.getenv("STRANDS_MODEL_ID", "command-a-03-2025"),
            "params": {"max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "8000"))},
        }

    elif provider == "github":
        return {
            "client_args": {
                "api_key": os.getenv("PAT_TOKEN", os.getenv("GITHUB_TOKEN")),
                "base_url": "https://models.github.ai/inference",
            },
            "model_id": os.getenv("STRANDS_MODEL_ID", "openai/o4-mini"),
            "params": {"max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "4000"))},
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_available_providers() -> list[str]:
    """Get list of available model providers.

    Returns:
        List of available model provider names.
    """
    return [
        "bedrock",
        "anthropic",
        "litellm",
        "llamaapi",
        "ollama",
        "openai",
        "writer",
        "cohere",
        "github",
    ]


def get_provider_info(provider: str) -> dict[str, Any]:
    """Get information about a specific model provider.

    Args:
        provider: Model provider name.

    Returns:
        Dictionary with provider information.
    """
    provider_info = {
        "bedrock": {
            "name": "Amazon Bedrock",
            "description": "Amazon's managed foundation model service",
            "default_model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "env_vars": [
                "STRANDS_MODEL_ID",
                "STRANDS_MAX_TOKENS",
                "AWS_PROFILE",
                "AWS_REGION",
            ],
        },
        "anthropic": {
            "name": "Anthropic",
            "description": "Direct access to Anthropic's Claude models",
            "default_model": "claude-sonnet-4-20250514",
            "env_vars": [
                "ANTHROPIC_API_KEY",
                "STRANDS_MODEL_ID",
                "STRANDS_MAX_TOKENS",
                "STRANDS_TEMPERATURE",
            ],
        },
        "litellm": {
            "name": "LiteLLM",
            "description": "Unified interface for multiple LLM providers",
            "default_model": "anthropic/claude-sonnet-4-20250514",
            "env_vars": [
                "LITELLM_API_KEY",
                "LITELLM_BASE_URL",
                "STRANDS_MODEL_ID",
                "STRANDS_MAX_TOKENS",
            ],
        },
        "llamaapi": {
            "name": "Llama API",
            "description": "Meta-hosted Llama model API service",
            "default_model": "llama3.1-405b",
            "env_vars": ["LLAMAAPI_API_KEY", "STRANDS_MODEL_ID", "STRANDS_MAX_TOKENS"],
        },
        "ollama": {
            "name": "Ollama",
            "description": "Local model inference server",
            "default_model": "llama3",
            "env_vars": ["OLLAMA_HOST", "STRANDS_MODEL_ID"],
        },
        "openai": {
            "name": "OpenAI",
            "description": "OpenAI's GPT models",
            "default_model": "o4-mini",
            "env_vars": ["OPENAI_API_KEY", "STRANDS_MODEL_ID", "STRANDS_MAX_TOKENS"],
        },
        "writer": {
            "name": "Writer",
            "description": "Writer models",
            "default_model": "palmyra-x5",
            "env_vars": ["WRITER_API_KEY", "STRANDS_MODEL_ID", "STRANDS_MAX_TOKENS"],
        },
        "cohere": {
            "name": "Cohere",
            "description": "Cohere models",
            "default_model": "command-a-03-2025",
            "env_vars": ["COHERE_API_KEY", "STRANDS_MODEL_ID", "STRANDS_MAX_TOKENS"],
        },
        "github": {
            "name": "GitHub",
            "description": "GitHub's model inference service",
            "default_model": "o4-mini",
            "env_vars": [
                "GITHUB_TOKEN",
                "PAT_TOKEN",
                "STRANDS_MODEL_ID",
                "STRANDS_MAX_TOKENS",
            ],
        },
    }

    return provider_info.get(provider, {"name": provider, "description": "Custom provider"})
