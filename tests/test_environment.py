"""
Tests for the environment tool using the Agent interface.
"""

import os

import pytest
from strands import Agent
from strands_tools import environment


@pytest.fixture
def agent():
    """Create an agent with the environment tool loaded."""
    return Agent(tools=[environment])


@pytest.fixture
def test_env_var():
    """Create and clean up a test environment variable."""
    var_name = "TEST_ENV_VAR"
    var_value = "test_value"
    os.environ[var_name] = var_value
    yield var_name, var_value
    # Clean up: remove test variable if it exists
    if var_name in os.environ:
        del os.environ[var_name]


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


def test_direct_list_with_prefix(agent, test_env_var):
    """Test listing environment variables with a specific prefix."""
    var_name, _ = test_env_var
    result = agent.tool.environment(action="list", prefix=var_name[:4])
    assert result["status"] == "success"
    # Verify our test variable is in the result
    assert var_name in extract_result_text(result)


def test_direct_get_existing_var(agent, test_env_var):
    """Test getting an existing environment variable."""
    var_name, var_value = test_env_var
    result = agent.tool.environment(action="get", name=var_name)
    assert result["status"] == "success"
    assert var_name in extract_result_text(result)
    assert var_value in extract_result_text(result)


def test_direct_get_nonexistent_var(agent):
    """Test getting a non-existent environment variable."""
    result = agent.tool.environment(action="get", name="NONEXISTENT_VAR_FOR_TEST")
    assert result["status"] == "error"
    assert "not found" in extract_result_text(result)


def test_direct_set_protected_var(agent, monkeypatch):
    """Test attempting to set a protected environment variable."""
    # Mock get_user_input to always return 'y' for confirmation
    monkeypatch.setattr("strands_tools.utils.user_input.get_user_input", lambda _: "y")

    # Try to modify PATH which is in PROTECTED_VARS
    result = agent.tool.environment(action="set", name="PATH", value="/bad/path")
    assert result["status"] == "error"
    # Verify the PATH was not changed to our bad value
    assert os.environ["PATH"] != "/bad/path"


@pytest.mark.skipif(os.name == "nt", reason="Need to mock console output for Windows (see #17)")
def test_direct_set_var_cancelled(agent, monkeypatch):
    """Test cancelling setting an environment variable."""
    # Mock get_user_input to return 'n' to cancel
    monkeypatch.setattr("strands_tools.utils.user_input.get_user_input", lambda _: "n")

    var_name = "CANCELLED_VAR"
    var_value = "cancelled_value"

    # Clean up in case the variable exists
    if var_name in os.environ:
        del os.environ[var_name]

    try:
        result = agent.tool.environment(action="set", name=var_name, value=var_value)
        assert result["status"] == "error"
        assert "cancelled" in extract_result_text(result).lower()
        # Verify variable was not set
        assert var_name not in os.environ
    finally:
        # Clean up
        if var_name in os.environ:
            del os.environ[var_name]


def test_direct_delete_nonexistent_var(agent, monkeypatch):
    """Test attempting to delete a non-existent variable."""
    # Mock get_user_input to always return 'y' for confirmation
    monkeypatch.setattr("strands_tools.utils.user_input.get_user_input", lambda _: "y")

    var_name = "NONEXISTENT_VAR_FOR_DELETE_TEST"

    # Make sure the variable doesn't exist
    if var_name in os.environ:
        del os.environ[var_name]

    result = agent.tool.environment(action="delete", name=var_name)
    assert result["status"] == "error"
    assert "not found" in extract_result_text(result).lower()


def test_direct_delete_protected_var(agent, monkeypatch):
    """Test attempting to delete a protected environment variable."""
    # Mock get_user_input to always return 'y' for confirmation
    monkeypatch.setattr("strands_tools.utils.user_input.get_user_input", lambda _: "y")

    # Try to delete PATH which is in PROTECTED_VARS
    original_path = os.environ.get("PATH", "")
    try:
        result = agent.tool.environment(action="delete", name="PATH")
        assert result["status"] == "error"
        # Verify PATH still exists
        assert "PATH" in os.environ
    finally:
        # Restore PATH if somehow it got deleted
        if "PATH" not in os.environ:
            os.environ["PATH"] = original_path


@pytest.mark.skipif(os.name == "nt", reason="Need to mock console output for Windows (see #17)")
def test_direct_delete_var_cancelled(agent, monkeypatch):
    """Test cancelling deletion of an environment variable."""
    # Mock get_user_input to return 'n' to cancel
    monkeypatch.setattr("strands_tools.utils.user_input.get_user_input", lambda _: "n")

    # Ensure DEV mode is disabled to force confirmation
    current_dev = os.environ.get("DEV", None)
    if current_dev:
        os.environ.pop("DEV")

    var_name = "CANCEL_DELETE_VAR"
    var_value = "cancel_delete_value"

    # Set up the variable
    os.environ[var_name] = var_value

    try:
        result = agent.tool.environment(action="delete", name=var_name)
        assert result["status"] == "error"
        assert "cancelled" in extract_result_text(result).lower()
        # Verify variable still exists
        assert var_name in os.environ
        assert os.environ[var_name] == var_value
    finally:
        # Clean up
        if var_name in os.environ:
            del os.environ[var_name]
        # Restore DEV mode if it was set
        if current_dev:
            os.environ["DEV"] = current_dev
        if var_name in os.environ:
            del os.environ[var_name]


def test_direct_validate_existing_var(agent, test_env_var):
    """Test validating an existing environment variable."""
    var_name, _ = test_env_var
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


def test_environment_dev_mode_delete(agent):
    """Test the environment tool in DEV mode with delete action."""
    # Set DEV mode
    original_dev = os.environ.get("DEV")
    os.environ["DEV"] = "true"

    var_name = "DEV_MODE_DELETE_VAR"
    var_value = "dev_mode_delete_value"

    try:
        # Set up the variable
        os.environ[var_name] = var_value

        result = agent.tool.environment(action="delete", name=var_name)
        assert result["status"] == "success"
        assert var_name not in os.environ
    finally:
        # Clean up
        if var_name in os.environ:
            del os.environ[var_name]

        # Restore original DEV value
        if original_dev is None:
            if "DEV" in os.environ:
                del os.environ["DEV"]
        else:
            os.environ["DEV"] = original_dev


def test_environment_dev_mode_protected_var(agent, monkeypatch):
    """Test that protected variables are still protected in DEV mode."""
    # Set DEV mode
    original_dev = os.environ.get("DEV")
    os.environ["DEV"] = "true"

    try:
        # Try to modify PATH which is protected
        result = agent.tool.environment(action="set", name="PATH", value="/bad/path")
        assert result["status"] == "error"
        # Verify PATH was not changed
        assert os.environ["PATH"] != "/bad/path"
    finally:
        # Restore original DEV value
        if original_dev is None:
            if "DEV" in os.environ:
                del os.environ["DEV"]
        else:
            os.environ["DEV"] = original_dev


def test_environment_masked_values(agent, test_env_var):
    """Test that sensitive values are masked in output."""
    # Create a sensitive looking variable
    sensitive_name = "TEST_TOKEN_SECRET"
    sensitive_value = "abcd1234efgh5678"
    os.environ[sensitive_name] = sensitive_value

    try:
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
    finally:
        # Clean up
        if sensitive_name in os.environ:
            del os.environ[sensitive_name]
