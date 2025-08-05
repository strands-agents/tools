"""
Tests for AWS utility functions.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.utils.aws_util import resolve_region


class TestResolveRegion:
    """Test the resolve_region function."""

    def test_explicit_region_provided(self):
        """Test that explicitly provided region is returned."""
        result = resolve_region("eu-west-1")
        assert result == "eu-west-1"

        result = resolve_region("ap-southeast-2")
        assert result == "ap-southeast-2"

    def test_explicit_region_various_formats(self):
        """Test various region name formats."""
        test_regions = [
            "us-east-1",
            "us-west-2", 
            "eu-central-1",
            "ap-northeast-1",
            "sa-east-1",
            "ca-central-1",
            "af-south-1",
            "me-south-1"
        ]
        
        for region in test_regions:
            result = resolve_region(region)
            assert result == region

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_boto3_session_region_available(self, mock_session_class):
        """Test using boto3 session region when available."""
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session_class.return_value = mock_session

        with patch.dict(os.environ, {}, clear=True):
            result = resolve_region()
            assert result == "us-east-1"

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_boto3_session_no_region(self, mock_session_class):
        """Test fallback when boto3 session has no region."""
        mock_session = MagicMock()
        mock_session.region_name = None
        mock_session_class.return_value = mock_session

        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
            result = resolve_region()
            assert result == "us-west-2"

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_boto3_session_exception(self, mock_session_class):
        """Test fallback when boto3 session creation raises exception."""
        mock_session_class.side_effect = Exception("Session creation failed")

        # Clear AWS_REGION env var if it exists
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_region()
            assert result == "us-west-2"  # DEFAULT_BEDROCK_REGION

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_boto3_session_various_exceptions(self, mock_session_class):
        """Test various exceptions during session creation."""
        exceptions = [
            Exception("General error"),
            RuntimeError("Runtime error"),
            ValueError("Value error"),
            ImportError("Import error"),
            AttributeError("Attribute error")
        ]
        
        for exception in exceptions:
            mock_session_class.side_effect = exception
            
            with patch.dict(os.environ, {}, clear=True):
                result = resolve_region()
                assert result == "us-west-2"

    def test_environment_variable_fallback(self):
        """Test using AWS_REGION environment variable."""
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region()
                assert result == "eu-west-1"

    def test_environment_variable_empty_string(self):
        """Test behavior with empty AWS_REGION environment variable."""
        with patch.dict(os.environ, {"AWS_REGION": ""}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region()
                # Empty string is falsy, should fall back to default
                assert result == "us-west-2"

    def test_default_region_fallback(self):
        """Test fallback to default region when nothing else available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region()
                assert result == "us-west-2"

    def test_empty_string_region_treated_as_none(self):
        """Test that empty string region is treated as None."""
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region("")
                # Empty string should be treated as None, so should fall back to env var
                assert result == "us-east-1"

    def test_none_region_explicit(self):
        """Test explicitly passing None as region."""
        with patch.dict(os.environ, {"AWS_REGION": "ap-southeast-1"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region(None)
                assert result == "ap-southeast-1"

    def test_resolution_hierarchy_complete(self):
        """Test the complete resolution hierarchy in order."""
        # Test 1: Explicit region wins over everything
        with patch.dict(os.environ, {"AWS_REGION": "env-region"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = "session-region"
                mock_session_class.return_value = mock_session

                result = resolve_region("explicit-region")
                assert result == "explicit-region"

        # Test 2: Session region wins over env var and default
        with patch.dict(os.environ, {"AWS_REGION": "env-region"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = "session-region"
                mock_session_class.return_value = mock_session

                result = resolve_region(None)
                assert result == "session-region"

        # Test 3: Env var wins over default when no session region
        with patch.dict(os.environ, {"AWS_REGION": "env-region"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region(None)
                assert result == "env-region"

        # Test 4: Default is used when nothing else available
        with patch.dict(os.environ, {}, clear=True):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region(None)
                assert result == "us-west-2"

    def test_whitespace_region_handling(self):
        """Test handling of regions with whitespace."""
        # Leading/trailing whitespace should be preserved (caller's responsibility to clean)
        result = resolve_region("  us-east-1  ")
        assert result == "  us-east-1  "
        
        result = resolve_region("\tus-west-2\n")
        assert result == "\tus-west-2\n"

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_session_region_with_whitespace(self, mock_session_class):
        """Test session region with whitespace."""
        mock_session = MagicMock()
        mock_session.region_name = "  us-east-1  "
        mock_session_class.return_value = mock_session

        with patch.dict(os.environ, {}, clear=True):
            result = resolve_region()
            assert result == "  us-east-1  "

    def test_environment_variable_with_whitespace(self):
        """Test environment variable with whitespace."""
        with patch.dict(os.environ, {"AWS_REGION": "  eu-west-1  "}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                result = resolve_region()
                assert result == "  eu-west-1  "

    def test_default_bedrock_region_import(self):
        """Test that DEFAULT_BEDROCK_REGION can be imported and has correct value."""
        from strands.models.bedrock import DEFAULT_BEDROCK_REGION
        assert DEFAULT_BEDROCK_REGION == "us-west-2"

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_session_creation_called_when_no_explicit_region(self, mock_session_class):
        """Test that boto3.Session is only called when no explicit region provided."""
        mock_session = MagicMock()
        mock_session.region_name = "session-region"
        mock_session_class.return_value = mock_session

        # When explicit region provided, should not call Session
        result = resolve_region("explicit-region")
        assert result == "explicit-region"
        mock_session_class.assert_not_called()

        # When no explicit region, should call Session
        result = resolve_region(None)
        assert result == "session-region"
        mock_session_class.assert_called_once()

    def test_multiple_calls_consistency(self):
        """Test that multiple calls with same parameters return consistent results."""
        with patch.dict(os.environ, {"AWS_REGION": "consistent-region"}):
            with patch("strands_tools.utils.aws_util.boto3.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session.region_name = None
                mock_session_class.return_value = mock_session

                # Multiple calls should return same result
                result1 = resolve_region()
                result2 = resolve_region()
                result3 = resolve_region()
                
                assert result1 == result2 == result3 == "consistent-region"
