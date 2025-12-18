"""
Tests for the environment tool using the Agent interface.
"""

import os
from unittest import mock

import pytest
from strands import Agent

from strands_tools import environment
from strands_tools.utils import user_input


@pytest.fixture
def agent():
    """Create an agent with the environment tool loaded."""
    return Agent(tools=[environment], load_tools_from_directory=False)


@pytest.fixture(autouse=True)
def get_user_input():
    with mock.patch.object(user_input, "get_user_input") as mocked_user_input:
        # By default all tests will return deny
        mocked_user_input.return_value = "n"
        yield mocked_user_input


@pytest.fixture(autouse=True)
def os_environment():
    mock_env = {}
    with mock.patch.object(os, "environ", mock_env):
        yield mock_env


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_direct_list_action(agent):
    """Test listing all environment variables."""
    result = agent.tool.environment(action="list")
    assert result["status"] == "success"
    # Just verify we got some sort of output since environment variables will differ
    assert len(extract_result_text(result)) > 0


def test_direct_list_with_prefix(agent, os_environment):
    """Test listing environment variables with a specific prefix."""
    var_name = "TEST_ENV_VAR"
    var_value = "test_value"
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="list", prefix=var_name[:4])
    assert result["status"] == "success"
    # Verify our test variable is in the result
    assert var_name in extract_result_text(result)


def test_direct_get_existing_var(agent, os_environment):
    """Test getting an existing environment variable."""
    var_name = "TEST_ENV_VAR"
    var_value = "test_value"
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="get", name=var_name)
    assert result["status"] == "success"
    assert var_name in extract_result_text(result)
    assert var_value in extract_result_text(result)


def test_direct_get_nonexistent_var(agent):
    """Test getting a non-existent environment variable."""
    result = agent.tool.environment(action="get", name="NONEXISTENT_VAR_FOR_TEST")
    assert result["status"] == "error"
    assert "not found" in extract_result_text(result)


def test_direct_set_protected_var(agent, os_environment):
    """Test attempting to set a protected environment variable."""
    os_environment["PATH"] = "/original/path"

    # Try to modify PATH which is in PROTECTED_VARS
    result = agent.tool.environment(action="set", name="PATH", value="/bad/path")
    assert result["status"] == "error"
    # Verify the PATH was not changed to our bad value
    assert os.environ["PATH"] != "/bad/path"


def test_direct_set_var_allowed(get_user_input, agent):
    """Test attempting to set a protected environment variable and allowing it."""
    get_user_input.return_value = "y"

    var_name = "CANCELLED_VAR"
    var_value = "cancelled_value"

    result = agent.tool.environment(action="set", name=var_name, value=var_value)
    assert result["status"] == "success"
    assert var_name in os.environ
    assert get_user_input.call_count == 1


def test_direct_set_var_cancelled(agent):
    var_name = "CANCELLED_VAR"
    var_value = "cancelled_value"

    result = agent.tool.environment(action="set", name=var_name, value=var_value)
    assert result["status"] == "error"
    assert "cancelled" in extract_result_text(result).lower()
    # Verify variable was not set
    assert var_name not in os.environ


def test_direct_delete_nonexistent_var(agent):
    """Test attempting to delete a non-existent variable."""
    var_name = "NONEXISTENT_VAR_FOR_DELETE_TEST"

    result = agent.tool.environment(action="delete", name=var_name)
    assert result["status"] == "error"
    assert "not found" in extract_result_text(result).lower()


def test_direct_delete_protected_var(agent, os_environment):
    """Test attempting to delete a protected environment variable."""
    # Try to delete PATH which is in PROTECTED_VARS
    unchanging_value = "/original/path"
    os_environment["PATH"] = unchanging_value

    result = agent.tool.environment(action="delete", name="PATH")
    assert result["status"] == "error"
    # Verify PATH still exists
    assert os_environment["PATH"] == unchanging_value


def test_direct_delete_var_cancelled(agent, os_environment):
    """Test cancelling deletion of an environment variable."""
    var_name = "CANCEL_DELETE_VAR"
    var_value = "cancel_delete_value"

    # Set up the variable
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="delete", name=var_name)
    assert result["status"] == "error"
    assert "cancelled" in extract_result_text(result).lower()
    # Verify variable still exists
    assert var_name in os.environ
    assert os_environment[var_name] == var_value


def test_direct_delete_var_allowed(agent, get_user_input, os_environment):
    """Test allowing deletion of an environment variable."""
    get_user_input.return_value = "y"

    var_name = "CANCEL_DELETE_VAR"
    var_value = "cancel_delete_value"

    # Set up the variable
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="delete", name=var_name)
    assert result["status"] == "success"
    assert "deleted environment variable" in extract_result_text(result).lower()
    assert os_environment.get(var_name) is None


def test_direct_validate_existing_var(agent, os_environment):
    """Test validating an existing environment variable."""
    var_name = "CANCEL_DELETE_VAR"
    var_value = "cancel_delete_value"

    # Set up the variable
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="validate", name=var_name)
    assert result["status"] == "success"
    assert "valid" in extract_result_text(result).lower()


def test_direct_validate_nonexistent_var(agent):
    """Test validating a non-existent environment variable."""
    result = agent.tool.environment(action="validate", name="NONEXISTENT_VAR_FOR_VALIDATION")
    assert result["status"] == "error"
    assert "not found" in extract_result_text(result).lower()


def test_direct_missing_parameters(agent):
    """Test providing incomplete parameters to the environment tool."""
    # Missing name for get action
    result = agent.tool.environment(action="get")
    assert result["status"] == "error"

    # Missing name and value for set action
    result = agent.tool.environment(action="set")
    assert result["status"] == "error"

    # Missing name for delete action
    result = agent.tool.environment(action="delete")
    assert result["status"] == "error"


def test_environment_dev_mode_delete(agent, os_environment):
    """Test the environment tool in BYPASS_TOOL_CONSENT mode with delete action."""
    # Set BYPASS_TOOL_CONSENT mode
    os_environment["BYPASS_TOOL_CONSENT"] = "true"

    var_name = "BYPASS_TOOL_CONSENT_MODE_DELETE_VAR"
    var_value = "dev_mode_delete_value"

    # Set up the variable
    os_environment[var_name] = var_value

    result = agent.tool.environment(action="delete", name=var_name)
    assert result["status"] == "success"
    assert var_name not in os_environment


def test_environment_dev_mode_protected_var(agent, os_environment):
    """Test that protected variables are still protected in BYPASS_TOOL_CONSENT mode."""
    # Set BYPASS_TOOL_CONSENT mode
    os_environment["BYPASS_TOOL_CONSENT"] = True

    unchanging_value = "/original/path"
    os_environment["PATH"] = unchanging_value

    # Try to modify PATH which is protected
    result = agent.tool.environment(action="set", name="PATH", value="/bad/path")
    assert result["status"] == "error"
    # Verify PATH was not changed
    assert os_environment != unchanging_value


def test_environment_masked_values(agent, os_environment):
    """Test that sensitive values are masked in output."""
    # Create a sensitive looking variable
    sensitive_name = "TEST_TOKEN_SECRET"
    sensitive_value = "abcd1234efgh5678"
    os_environment[sensitive_name] = sensitive_value

    # Test with masking enabled (default)
    result = agent.tool.environment(action="get", name=sensitive_name)
    assert result["status"] == "success"
    # The full value should not appear in the output
    assert sensitive_value not in extract_result_text(result)

    # Test with masking disabled
    result = agent.tool.environment(action="get", name=sensitive_name, masked=False)
    assert result["status"] == "success"
    # Now the full value should appear
    assert sensitive_value in extract_result_text(result)
