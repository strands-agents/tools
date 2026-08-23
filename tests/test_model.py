"""Tests for the model provider registry (strands_tools.utils.models.model)."""

from unittest.mock import patch

from strands_tools.utils.models.model import (
    create_model,
    get_available_providers,
    get_provider_config,
    get_provider_info,
)


def test_get_provider_config_orcarouter_defaults():
    """OrcaRouter provider should default to the OrcaRouter gateway base URL and auto model."""
    config = get_provider_config("orcarouter")
    assert config["client_args"]["base_url"] == "https://api.orcarouter.ai/v1"
    assert config["model_id"] == "orcarouter/auto"


def test_get_provider_config_orcarouter_env_override(monkeypatch):
    """ORCAROUTER_API_KEY and ORCAROUTER_BASE_URL should be honored."""
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    monkeypatch.setenv("ORCAROUTER_BASE_URL", "https://custom.example/v1")
    config = get_provider_config("orcarouter")
    assert config["client_args"]["api_key"] == "sk-orca-test"
    assert config["client_args"]["base_url"] == "https://custom.example/v1"


def test_get_available_providers_includes_orcarouter():
    assert "orcarouter" in get_available_providers()


def test_get_provider_info_orcarouter():
    info = get_provider_info("orcarouter")
    assert info["name"] == "OrcaRouter"
    assert "ORCAROUTER_API_KEY" in info["env_vars"]


def test_create_model_orcarouter_uses_openai_model():
    """The orcarouter provider is OpenAI-compatible and should build an OpenAIModel."""
    with patch("strands.models.openai.OpenAIModel") as mock_cls:
        create_model("orcarouter", {"model_id": "orcarouter/auto"})
    mock_cls.assert_called_once_with(model_id="orcarouter/auto")
