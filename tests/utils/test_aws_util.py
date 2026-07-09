"""
Tests for AWS utility functions.
"""

import os
from unittest.mock import MagicMock, patch

from strands_tools.utils.aws_util import DEFAULT_BEDROCK_REGION, resolve_region


class TestResolveRegion:
    """Test the resolve_region function."""

    def test_explicit_region_provided(self):
        """Test that explicitly provided region is returned."""
        result = resolve_region("eu-west-1")
        assert result == "eu-west-1"

        result = resolve_region("ap-southeast-2")
        assert result == "ap-southeast-2"

    @patch("strands_tools.utils.aws_util.boto3.Session")
    def test_boto3_session_exception(self, mock_session_class):
        """Test fallback when boto3 session creation raises exception."""
        mock_session_class.side_effect = Exception("Session creation failed")

        # Clear AWS_REGION env var if it exists
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_region()
            assert result == DEFAULT_BEDROCK_REGION

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
                assert result == DEFAULT_BEDROCK_REGION

    def test_default_bedrock_region_constant(self):
        """Test that the default region constant is correct."""
        assert DEFAULT_BEDROCK_REGION == "us-west-2"
