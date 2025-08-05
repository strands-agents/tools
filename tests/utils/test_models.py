"""
Tests for model utility functions.

These tests focus on the basic functionality of the model utility functions
without requiring the actual model dependencies to be installed.
"""

import sys
from unittest.mock import Mock, patch

import pytest


class TestModelUtilities:
    """Test model utility functions."""

    def test_anthropic_instance_function_exists(self):
        """Test that the anthropic instance function exists and is callable."""
        # Mock the dependencies to avoid import errors
        with patch.dict('sys.modules', {
            'strands.models.anthropic': Mock(),
            'anthropic': Mock()
        }):
            from strands_tools.utils.models import anthropic
            assert hasattr(anthropic, 'instance')
            assert callable(anthropic.instance)

    def test_bedrock_instance_function_exists(self):
        """Test that the bedrock instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models': Mock(),
            'botocore.config': Mock()
        }):
            from strands_tools.utils.models import bedrock
            assert hasattr(bedrock, 'instance')
            assert callable(bedrock.instance)

    def test_litellm_instance_function_exists(self):
        """Test that the litellm instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models.litellm': Mock()
        }):
            from strands_tools.utils.models import litellm
            assert hasattr(litellm, 'instance')
            assert callable(litellm.instance)

    def test_llamaapi_instance_function_exists(self):
        """Test that the llamaapi instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models.llamaapi': Mock()
        }):
            from strands_tools.utils.models import llamaapi
            assert hasattr(llamaapi, 'instance')
            assert callable(llamaapi.instance)

    def test_ollama_instance_function_exists(self):
        """Test that the ollama instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models.ollama': Mock()
        }):
            from strands_tools.utils.models import ollama
            assert hasattr(ollama, 'instance')
            assert callable(ollama.instance)

    def test_openai_instance_function_exists(self):
        """Test that the openai instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models.openai': Mock()
        }):
            from strands_tools.utils.models import openai
            assert hasattr(openai, 'instance')
            assert callable(openai.instance)

    def test_writer_instance_function_exists(self):
        """Test that the writer instance function exists and is callable."""
        with patch.dict('sys.modules', {
            'strands.models.writer': Mock()
        }):
            from strands_tools.utils.models import writer
            assert hasattr(writer, 'instance')
            assert callable(writer.instance)


class TestAnthropicModel:
    """Test Anthropic model utility with mocked dependencies."""

    @patch.dict('sys.modules', {
        'strands.models.anthropic': Mock(),
        'anthropic': Mock()
    })
    def test_instance_creation(self):
        """Test creating an Anthropic model instance."""
        # Import after patching
        from strands_tools.utils.models import anthropic
        
        # Mock the AnthropicModel class
        with patch('strands_tools.utils.models.anthropic.AnthropicModel') as mock_anthropic_model:
            mock_model = Mock()
            mock_anthropic_model.return_value = mock_model
            
            config = {"model": "claude-3-sonnet", "api_key": "test-key"}
            result = anthropic.instance(**config)
            
            mock_anthropic_model.assert_called_once_with(**config)
            assert result == mock_model

    @patch.dict('sys.modules', {
        'strands.models.anthropic': Mock(),
        'anthropic': Mock()
    })
    def test_instance_no_config(self):
        """Test creating an Anthropic model instance with no config."""
        from strands_tools.utils.models import anthropic
        
        with patch('strands_tools.utils.models.anthropic.AnthropicModel') as mock_anthropic_model:
            mock_model = Mock()
            mock_anthropic_model.return_value = mock_model
            
            result = anthropic.instance()
            
            mock_anthropic_model.assert_called_once_with()
            assert result == mock_model


class TestBedrockModel:
    """Test Bedrock model utility with mocked dependencies."""

    @patch.dict('sys.modules', {
        'strands.models': Mock(),
        'botocore.config': Mock()
    })
    def test_instance_creation(self):
        """Test creating a Bedrock model instance."""
        from strands_tools.utils.models import bedrock
        
        with patch('strands_tools.utils.models.bedrock.BedrockModel') as mock_bedrock_model:
            mock_model = Mock()
            mock_bedrock_model.return_value = mock_model
            
            config = {"model": "anthropic.claude-3-sonnet", "region": "us-east-1"}
            result = bedrock.instance(**config)
            
            mock_bedrock_model.assert_called_once_with(**config)
            assert result == mock_model

    @patch.dict('sys.modules', {
        'strands.models': Mock(),
        'botocore.config': Mock()
    })
    def test_instance_with_boto_config_dict(self):
        """Test creating a Bedrock model instance with boto config as dict."""
        from strands_tools.utils.models import bedrock
        
        with patch('strands_tools.utils.models.bedrock.BedrockModel') as mock_bedrock_model, \
             patch('strands_tools.utils.models.bedrock.BotocoreConfig') as mock_botocore_config:
            
            mock_model = Mock()
            mock_bedrock_model.return_value = mock_model
            mock_boto_config = Mock()
            mock_botocore_config.return_value = mock_boto_config
            
            boto_config_dict = {"region_name": "us-west-2", "retries": {"max_attempts": 3}}
            config = {
                "model": "anthropic.claude-3-sonnet",
                "boto_client_config": boto_config_dict
            }
            
            result = bedrock.instance(**config)
            
            mock_botocore_config.assert_called_once_with(**boto_config_dict)
            expected_config = config.copy()
            expected_config["boto_client_config"] = mock_boto_config
            mock_bedrock_model.assert_called_once_with(**expected_config)
            assert result == mock_model