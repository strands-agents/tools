"""
Core tests for model utility functions that don't require external dependencies.
"""

import json
import os
import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest
from botocore.config import Config


class TestModelConfiguration:
    """Test model configuration loading and defaults."""

    def setup_method(self):
        """Reset environment variables before each test."""
        self.env_vars_to_clear = [
            "STRANDS_MODEL_ID", "STRANDS_MAX_TOKENS", "STRANDS_BOTO_READ_TIMEOUT",
            "STRANDS_BOTO_CONNECT_TIMEOUT", "STRANDS_BOTO_MAX_ATTEMPTS",
            "STRANDS_ADDITIONAL_REQUEST_FIELDS", "STRANDS_ANTHROPIC_BETA",
            "STRANDS_THINKING_TYPE", "STRANDS_BUDGET_TOKENS", "STRANDS_CACHE_TOOLS",
            "STRANDS_CACHE_PROMPT", "STRANDS_PROVIDER", "ANTHROPIC_API_KEY",
            "LITELLM_API_KEY", "LITELLM_BASE_URL", "LLAMAAPI_API_KEY",
            "OLLAMA_HOST", "OPENAI_API_KEY", "WRITER_API_KEY", "COHERE_API_KEY",
            "PAT_TOKEN", "GITHUB_TOKEN", "STRANDS_TEMPERATURE"
        ]
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_default_model_config_basic(self):
        """Test default model configuration with no environment variables."""
        # Reload the module to ensure clean state
        import importlib
        from strands_tools.utils.models import model
        importlib.reload(model)
        
        from strands_tools.utils.models.model import DEFAULT_MODEL_CONFIG
        
        assert DEFAULT_MODEL_CONFIG["model_id"] == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        assert DEFAULT_MODEL_CONFIG["max_tokens"] == 10000
        assert isinstance(DEFAULT_MODEL_CONFIG["boto_client_config"], Config)
        assert DEFAULT_MODEL_CONFIG["additional_request_fields"] == {}
        assert DEFAULT_MODEL_CONFIG["cache_tools"] == "default"
        assert DEFAULT_MODEL_CONFIG["cache_prompt"] == "default"

    def test_default_model_config_with_env_vars(self):
        """Test default model configuration with environment variables."""
        with patch.dict(os.environ, {
            "STRANDS_MODEL_ID": "custom-model",
            "STRANDS_MAX_TOKENS": "5000",
            "STRANDS_BOTO_READ_TIMEOUT": "600",
            "STRANDS_BOTO_CONNECT_TIMEOUT": "300",
            "STRANDS_BOTO_MAX_ATTEMPTS": "5",
            "STRANDS_CACHE_TOOLS": "ephemeral",
            "STRANDS_CACHE_PROMPT": "ephemeral"
        }):
            # Re-import to get updated config
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            assert config["model_id"] == "custom-model"
            assert config["max_tokens"] == 5000
            assert config["cache_tools"] == "ephemeral"
            assert config["cache_prompt"] == "ephemeral"

    def test_additional_request_fields_parsing(self):
        """Test parsing of additional request fields from environment."""
        with patch.dict(os.environ, {
            "STRANDS_ADDITIONAL_REQUEST_FIELDS": '{"temperature": 0.7, "top_p": 0.9}'
        }):
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            assert config["additional_request_fields"]["temperature"] == 0.7
            assert config["additional_request_fields"]["top_p"] == 0.9

    def test_additional_request_fields_invalid_json(self):
        """Test handling of invalid JSON in additional request fields."""
        with patch.dict(os.environ, {
            "STRANDS_ADDITIONAL_REQUEST_FIELDS": "invalid-json"
        }):
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            assert config["additional_request_fields"] == {}

    def test_anthropic_beta_features(self):
        """Test parsing of Anthropic beta features."""
        with patch.dict(os.environ, {
            "STRANDS_ANTHROPIC_BETA": "feature1,feature2,feature3"
        }):
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            assert config["additional_request_fields"]["anthropic_beta"] == ["feature1", "feature2", "feature3"]

    def test_thinking_configuration(self):
        """Test thinking configuration setup."""
        with patch.dict(os.environ, {
            "STRANDS_THINKING_TYPE": "reasoning",
            "STRANDS_BUDGET_TOKENS": "1000"
        }):
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            thinking_config = config["additional_request_fields"]["thinking"]
            assert thinking_config["type"] == "reasoning"
            assert thinking_config["budget_tokens"] == 1000

    def test_thinking_configuration_no_budget(self):
        """Test thinking configuration without budget tokens."""
        with patch.dict(os.environ, {
            "STRANDS_THINKING_TYPE": "reasoning"
        }):
            import importlib
            from strands_tools.utils.models import model
            importlib.reload(model)
            
            config = model.DEFAULT_MODEL_CONFIG
            thinking_config = config["additional_request_fields"]["thinking"]
            assert thinking_config["type"] == "reasoning"
            assert "budget_tokens" not in thinking_config


class TestLoadPath:
    """Test the load_path function."""

    def test_load_path_cwd_models_exists(self):
        """Test loading path when .models directory exists in CWD."""
        from strands_tools.utils.models.model import load_path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .models directory and file
            models_dir = pathlib.Path(temp_dir) / ".models"
            models_dir.mkdir()
            model_file = models_dir / "custom.py"
            model_file.write_text("# Custom model")
            
            with patch("pathlib.Path.cwd", return_value=pathlib.Path(temp_dir)):
                result = load_path("custom")
                assert result == model_file
                assert result.exists()

    def test_load_path_builtin_models(self):
        """Test loading path from built-in models directory."""
        from strands_tools.utils.models.model import load_path
        
        # Mock the built-in path to exist
        with patch("pathlib.Path.exists") as mock_exists:
            # First call (CWD) returns False, second call (built-in) returns True
            mock_exists.side_effect = [False, True]
            
            result = load_path("bedrock")
            expected_path = pathlib.Path(__file__).parent.parent.parent / "src" / "strands_tools" / "utils" / "models" / ".." / "models" / "bedrock.py"
            # Just check that it's a Path object with the right name
            assert isinstance(result, pathlib.Path)
            assert result.name == "bedrock.py"

    def test_load_path_not_found(self):
        """Test loading path when model doesn't exist."""
        from strands_tools.utils.models.model import load_path
        
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ImportError, match="model_provider=<nonexistent> | does not exist"):
                load_path("nonexistent")


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_empty_string(self):
        """Test loading config with empty string returns default."""
        from strands_tools.utils.models.model import load_config, DEFAULT_MODEL_CONFIG
        
        result = load_config("")
        assert result == DEFAULT_MODEL_CONFIG

    def test_load_config_empty_json(self):
        """Test loading config with empty JSON returns default."""
        from strands_tools.utils.models.model import load_config, DEFAULT_MODEL_CONFIG
        
        result = load_config("{}")
        assert result == DEFAULT_MODEL_CONFIG

    def test_load_config_json_string(self):
        """Test loading config from JSON string."""
        from strands_tools.utils.models.model import load_config
        
        config_json = '{"model_id": "test-model", "max_tokens": 2000}'
        result = load_config(config_json)
        
        assert result["model_id"] == "test-model"
        assert result["max_tokens"] == 2000

    def test_load_config_json_file(self):
        """Test loading config from JSON file."""
        from strands_tools.utils.models.model import load_config
        
        config_data = {"model_id": "file-model", "max_tokens": 3000}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            assert result["model_id"] == "file-model"
            assert result["max_tokens"] == 3000
        finally:
            os.unlink(temp_file)


class TestLoadModel:
    """Test the load_model function."""

    def test_load_model_success(self):
        """Test successful model loading."""
        from strands_tools.utils.models.model import load_model
        
        # Create a temporary module file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def instance(**config):
    return f"Model with config: {config}"
""")
            temp_file = pathlib.Path(f.name)
        
        try:
            config = {"model_id": "test", "max_tokens": 1000}
            result = load_model(temp_file, config)
            assert result == f"Model with config: {config}"
        finally:
            os.unlink(temp_file)

    def test_load_model_with_mock(self):
        """Test load_model with mocked module loading."""
        from strands_tools.utils.models.model import load_model
        
        mock_module = Mock()
        mock_module.instance.return_value = "mocked_model"
        
        with patch("importlib.util.spec_from_file_location") as mock_spec_from_file, \
             patch("importlib.util.module_from_spec") as mock_module_from_spec:
            
            mock_spec = Mock()
            mock_loader = Mock()
            mock_spec.loader = mock_loader
            mock_spec_from_file.return_value = mock_spec
            mock_module_from_spec.return_value = mock_module
            
            path = pathlib.Path("test_model.py")
            config = {"test": "config"}
            
            result = load_model(path, config)
            
            mock_spec_from_file.assert_called_once_with("test_model", str(path))
            mock_module_from_spec.assert_called_once_with(mock_spec)
            mock_loader.exec_module.assert_called_once_with(mock_module)
            mock_module.instance.assert_called_once_with(**config)
            assert result == "mocked_model"


class TestGetProviderConfig:
    """Test the get_provider_config function."""

    def test_get_provider_config_bedrock(self):
        """Test getting bedrock provider config."""
        from strands_tools.utils.models.model import get_provider_config
        
        with patch.dict(os.environ, {
            "STRANDS_MODEL_ID": "custom-bedrock-model",
            "STRANDS_MAX_TOKENS": "8000",
            "STRANDS_CACHE_PROMPT": "ephemeral",
            "STRANDS_CACHE_TOOLS": "ephemeral"
        }):
            config = get_provider_config("bedrock")
            
            assert config["model_id"] == "custom-bedrock-model"
            assert config["max_tokens"] == 8000
            assert config["cache_prompt"] == "ephemeral"
            assert config["cache_tools"] == "ephemeral"
            assert isinstance(config["boto_client_config"], Config)

    def test_get_provider_config_anthropic(self):
        """Test getting anthropic provider config."""
        from strands_tools.utils.models.model import get_provider_config
        
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "STRANDS_MODEL_ID": "claude-3-opus",
            "STRANDS_MAX_TOKENS": "4000",
            "STRANDS_TEMPERATURE": "0.5"
        }):
            config = get_provider_config("anthropic")
            
            assert config["client_args"]["api_key"] == "test-key"
            assert config["model_id"] == "claude-3-opus"
            assert config["max_tokens"] == 4000
            assert config["params"]["temperature"] == 0.5

    def test_get_provider_config_litellm(self):
        """Test getting litellm provider config."""
        from strands_tools.utils.models.model import get_provider_config
        
        with patch.dict(os.environ, {
            "LITELLM_API_KEY": "litellm-key",
            "LITELLM_BASE_URL": "https://api.litellm.ai",
            "STRANDS_MODEL_ID": "gpt-4",
            "STRANDS_MAX_TOKENS": "2000",
            "STRANDS_TEMPERATURE": "0.8"
        }):
            config = get_provider_config("litellm")
            
            assert config["client_args"]["api_key"] == "litellm-key"
            assert config["client_args"]["base_url"] == "https://api.litellm.ai"
            assert config["model_id"] == "gpt-4"
            assert config["params"]["max_tokens"] == 2000
            assert config["params"]["temperature"] == 0.8

    def test_get_provider_config_litellm_no_base_url(self):
        """Test getting litellm provider config without base URL."""
        from strands_tools.utils.models.model import get_provider_config
        
        with patch.dict(os.environ, {"LITELLM_API_KEY": "litellm-key"}):
            config = get_provider_config("litellm")
            
            assert config["client_args"]["api_key"] == "litellm-key"
            assert "base_url" not in config["client_args"]

    def test_get_provider_config_unknown(self):
        """Test getting config for unknown provider."""
        from strands_tools.utils.models.model import get_provider_config
        
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            get_provider_config("unknown")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        from strands_tools.utils.models.model import get_available_providers
        
        providers = get_available_providers()
        expected_providers = [
            "bedrock", "anthropic", "litellm", "llamaapi", "ollama",
            "openai", "writer", "cohere", "github"
        ]
        
        assert providers == expected_providers
        assert len(providers) == 9

    def test_get_provider_info_bedrock(self):
        """Test getting provider info for bedrock."""
        from strands_tools.utils.models.model import get_provider_info
        
        info = get_provider_info("bedrock")
        
        assert info["name"] == "Amazon Bedrock"
        assert "Amazon's managed foundation model service" in info["description"]
        assert info["default_model"] == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        assert "STRANDS_MODEL_ID" in info["env_vars"]
        assert "AWS_PROFILE" in info["env_vars"]

    def test_get_provider_info_anthropic(self):
        """Test getting provider info for anthropic."""
        from strands_tools.utils.models.model import get_provider_info
        
        info = get_provider_info("anthropic")
        
        assert info["name"] == "Anthropic"
        assert "Direct access to Anthropic's Claude models" in info["description"]
        assert info["default_model"] == "claude-sonnet-4-20250514"
        assert "ANTHROPIC_API_KEY" in info["env_vars"]

    def test_get_provider_info_unknown(self):
        """Test getting provider info for unknown provider."""
        from strands_tools.utils.models.model import get_provider_info
        
        info = get_provider_info("unknown")
        
        assert info["name"] == "unknown"
        assert info["description"] == "Custom provider"


class TestCreateModelBasic:
    """Test basic create_model functionality without external dependencies."""

    def test_create_model_bedrock_default(self):
        """Test creating bedrock model with default provider."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands.models.bedrock.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = "bedrock_model"
            
            result = create_model()
            
            mock_bedrock.assert_called_once()
            assert result == "bedrock_model"

    def test_create_model_openai(self):
        """Test creating openai model."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = "openai_model"
            
            result = create_model("openai")
            
            mock_openai.assert_called_once()
            assert result == "openai_model"

    def test_create_model_cohere(self):
        """Test creating cohere model (uses OpenAI interface)."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = "cohere_model"
            
            result = create_model("cohere")
            
            mock_openai.assert_called_once()
            assert result == "cohere_model"

    def test_create_model_github(self):
        """Test creating github model (uses OpenAI interface)."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = "github_model"
            
            result = create_model("github")
            
            mock_openai.assert_called_once()
            assert result == "github_model"

    def test_create_model_custom_provider(self):
        """Test creating custom model provider."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands_tools.utils.models.model.load_path") as mock_load_path, \
             patch("strands_tools.utils.models.model.load_model") as mock_load_model:
            
            mock_path = pathlib.Path("custom.py")
            mock_load_path.return_value = mock_path
            mock_load_model.return_value = "custom_model"
            
            config = {"test": "config"}
            result = create_model("custom", config)
            
            mock_load_path.assert_called_once_with("custom")
            mock_load_model.assert_called_once_with(mock_path, config)
            assert result == "custom_model"

    def test_create_model_unknown_provider(self):
        """Test creating model with unknown provider."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands_tools.utils.models.model.load_path", side_effect=ImportError):
            with pytest.raises(ValueError, match="Unknown provider: unknown"):
                create_model("unknown")

    def test_create_model_with_custom_config(self):
        """Test creating model with custom config."""
        from strands_tools.utils.models.model import create_model
        
        with patch("strands.models.bedrock.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = "bedrock_model"
            
            custom_config = {"model_id": "custom", "max_tokens": 5000}
            result = create_model("bedrock", custom_config)
            
            mock_bedrock.assert_called_once_with(**custom_config)
            assert result == "bedrock_model"

    def test_load_config_and_create_model(self):
        """Test loading config from JSON and creating model."""
        from strands_tools.utils.models.model import load_config, create_model
        
        config_json = '{"model_id": "test-model", "max_tokens": 2000}'
        
        with patch("strands.models.bedrock.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = "bedrock_model"
            
            config = load_config(config_json)
            result = create_model("bedrock", config)
            
            mock_bedrock.assert_called_once_with(model_id="test-model", max_tokens=2000)
            assert result == "bedrock_model"