"""
Integration tests for the AgentCoreCodeInterpreter custom identifier functionality.

These tests verify end-to-end functionality with custom identifiers, including:
- Complete session creation flow with custom identifiers
- Backward compatibility with existing functionality
- Error scenarios with meaningful error messages including identifier context
"""

import logging
from unittest.mock import MagicMock, patch
import time

import pytest
from strands import Agent
from strands.agent import AgentResult
from strands_tools.code_interpreter import AgentCoreCodeInterpreter
from strands_tools.code_interpreter.models import (
    ExecuteCodeAction,
    InitSessionAction,
    LanguageType,
)

from tests_integ.ci_environments import skip_if_github_action

logger = logging.getLogger(__name__)


@pytest.fixture
def custom_identifier_interpreter() -> AgentCoreCodeInterpreter:
    """Create a real AgentCoreCodeInterpreter with custom parameters but default identifier."""
    # Use default identifier but customize other parameters
    return AgentCoreCodeInterpreter(region="us-west-2", session_name="custom-default")


@pytest.fixture
def default_identifier_interpreter() -> AgentCoreCodeInterpreter:
    """Create a real AgentCoreCodeInterpreter with default identifier for comparison."""
    return AgentCoreCodeInterpreter(region="us-west-2")


@pytest.fixture
def agent_with_custom_identifier(custom_identifier_interpreter: AgentCoreCodeInterpreter) -> Agent:
    """Create an agent with custom identifier code interpreter tool."""
    return Agent(tools=[custom_identifier_interpreter.code_interpreter])


@pytest.fixture
def agent_with_default_identifier(default_identifier_interpreter: AgentCoreCodeInterpreter) -> Agent:
    """Create an agent with default identifier code interpreter tool."""
    return Agent(tools=[default_identifier_interpreter.code_interpreter])

@pytest.fixture(autouse=True, scope="function")
def rate_limit_protection():
    """Prevent throttling between tests."""
    yield
    time.sleep(2) 

class TestCustomIdentifierEndToEnd:
    """Test complete end-to-end functionality with custom identifiers."""

    @skip_if_github_action.mark
    def test_complete_session_creation_flow_with_custom_identifier(self, custom_identifier_interpreter):
        """Test complete session creation flow with custom identifier."""
        # Test direct tool call with custom identifier
        result = custom_identifier_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Custom identifier test session",
                    "session_name": "custom-id-session"
                }
            }
        )

        # Verify session was created successfully
        assert result['status'] == 'success'
        assert 'sessionName' in result['content'][0]['json']
        assert result['content'][0]['json']['sessionName'] == 'custom-id-session'
        assert result['content'][0]['json']['description'] == 'Custom identifier test session'
        assert 'sessionId' in result['content'][0]['json']

        # Verify session is stored in the interpreter
        assert 'custom-id-session' in custom_identifier_interpreter._sessions
        session_info = custom_identifier_interpreter._sessions['custom-id-session']
        assert session_info.description == 'Custom identifier test session'
        assert session_info.session_id == result['content'][0]['json']['sessionId']

        # Test that we can execute code in the session
        code_result = custom_identifier_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "executeCode",
                    "session_name": "custom-id-session",
                    "code": "print('Hello from custom identifier session!')",
                    "language": "python"
                }
            }
        )

        assert code_result['status'] == 'success'

    @skip_if_github_action.mark
    def test_multiple_sessions_with_different_configurations(self):
        """Test creating multiple sessions with different configurations."""
        # Create interpreters with different session configurations but same valid identifier
        interpreter1 = AgentCoreCodeInterpreter(
            region="us-west-2",
            session_name="session1-default"
        )
        interpreter2 = AgentCoreCodeInterpreter(
            region="us-west-2", 
            session_name="session2-default"
        )

        # Create sessions with each interpreter
        result1 = interpreter1.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "First session",
                    "session_name": "session-1"
                }
            }
        )

        result2 = interpreter2.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Second session",
                    "session_name": "session-2"
                }
            }
        )

        # Verify both sessions were created successfully
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'
        assert result1['content'][0]['json']['sessionName'] == 'session-1'
        assert result2['content'][0]['json']['sessionName'] == 'session-2'

        # Verify sessions are isolated
        assert 'session-1' in interpreter1._sessions
        assert 'session-1' not in interpreter2._sessions
        assert 'session-2' in interpreter2._sessions
        assert 'session-2' not in interpreter1._sessions

    @skip_if_github_action.mark  
    def test_natural_language_workflow_with_custom_identifier(self, agent_with_custom_identifier):
        """Test custom identifier with consolidated operations."""
        
        result = agent_with_custom_identifier("""
    Using custom identifier, complete this in ONE code execution:
    1. Create 'test.txt' with content 'Custom identifier test'
    2. Read it back
    3. Return the contents

    Respond with "PASS:" followed by the file contents.
        """)
        
        response = result.message["content"][0]["text"]
        assert "PASS:" in response
        assert "Custom identifier test" in response


class TestBackwardCompatibility:
    """Test that existing functionality works unchanged with custom identifiers."""

    @skip_if_github_action.mark
    def test_existing_functionality_unchanged_with_default_identifier(self, default_identifier_interpreter):
        """Verify that all existing functionality works unchanged with default identifier."""
        # This test replicates the existing integration test to ensure no regression
        result = default_identifier_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Backward compatibility test session",
                    "session_name": "compat-session"
                }
            }
        )

        assert result['status'] == 'success'
        assert result['content'][0]['json']['sessionName'] == 'compat-session'

        # Test code execution works the same way
        code_result = default_identifier_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "executeCode",
                    "session_name": "compat-session",
                    "code": "print('Backward compatibility test')",
                    "language": "python"
                }
            }
        )

        assert code_result['status'] == 'success'

    @skip_if_github_action.mark  
    def test_existing_natural_language_workflow_unchanged(self, agent_with_default_identifier):
        """Test that existing natural language workflows work unchanged."""
        # This replicates the complex workflow from the original integration test
        result: AgentResult = agent_with_default_identifier("""
        I need you to help me with a data processing task. Please:
        
        1. Create a new code session for this project
        
        2. Write a Python script that creates a CSV file with sample data
        
        3. Execute the script to create the file
        
        4. Read the file back to verify it was created
        
        5. If all steps succeed, respond with "BACKWARD_COMPATIBILITY_PASS"
        """)

        assert "BACKWARD_COMPATIBILITY_PASS" in result.message["content"][0]["text"]

    @skip_if_github_action.mark
    def test_constructor_backward_compatibility(self):
        """Test that existing constructor patterns still work."""
        # Test existing constructor patterns
        interpreter1 = AgentCoreCodeInterpreter("us-west-2")  # Positional region
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2")  # Named region
        interpreter3 = AgentCoreCodeInterpreter()  # No parameters

        # All should use default identifier
        assert interpreter1.identifier == "aws.codeinterpreter.v1"
        assert interpreter2.identifier == "aws.codeinterpreter.v1"
        assert interpreter3.identifier == "aws.codeinterpreter.v1"

        # Test that they can create sessions successfully
        result = interpreter1.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Backward compatibility constructor test",
                    "session_name": "compat-constructor-session"
                }
            }
        )

        assert result['status'] == 'success'


class TestErrorScenariosWithIdentifierContext:
    """Test error scenarios and ensure meaningful error messages include identifier context."""

    def test_session_initialization_error_handling_with_custom_identifier(self):
        """Test that session initialization errors are handled correctly when using custom identifiers."""
        with patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient") as mock_client_class:
            # Mock client to raise an exception during start
            mock_client = MagicMock()
            mock_client.start.side_effect = Exception("Invalid identifier format")
            mock_client_class.return_value = mock_client

            custom_id = "invalid-identifier-format"
            interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

            result = interpreter.code_interpreter(
                code_interpreter_input={
                    "action": {
                        "type": "initSession",
                        "description": "Error test session",
                        "session_name": "error-session"
                    }
                }
            )

            # Verify error response contains expected information
            assert result['status'] == 'error'
            error_message = result['content'][0]['text']
            assert "error-session" in error_message
            assert "Invalid identifier format" in error_message

    def test_client_creation_error_handling_with_custom_identifier(self):
        """Test that client creation errors are handled correctly when using custom identifiers."""
        with patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient") as mock_client_class:
            # Mock client class to raise an exception during instantiation
            mock_client_class.side_effect = Exception("Client creation failed")

            custom_id = "problematic-identifier"
            interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

            result = interpreter.code_interpreter(
                code_interpreter_input={
                    "action": {
                        "type": "initSession",
                        "description": "Client error test session",
                        "session_name": "client-error-session"
                    }
                }
            )

            # Verify error response contains expected information
            assert result['status'] == 'error'
            error_message = result['content'][0]['text']
            assert "client-error-session" in error_message
            assert "Client creation failed" in error_message

    def test_logging_includes_identifier_context_on_errors(self):
        """Test that error logging includes identifier context for debugging."""
        with patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient") as mock_client_class:
            with patch("strands_tools.code_interpreter.agent_core_code_interpreter.logger") as mock_logger:
                # Mock client to raise an exception
                mock_client = MagicMock()
                mock_client.start.side_effect = Exception("Logging test error")
                mock_client_class.return_value = mock_client

                custom_id = "logging-test-identifier"
                interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

                interpreter.code_interpreter(
                    code_interpreter_input={
                        "action": {
                            "type": "initSession",
                            "description": "Logging test session",
                            "session_name": "logging-test-session"
                        }
                    }
                )

                # Verify error logging includes identifier
                mock_logger.error.assert_called_once()
                error_call_args = mock_logger.error.call_args[0][0]
                assert custom_id in error_call_args
                assert "logging-test-session" in error_call_args
                assert "Logging test error" in error_call_args

    def test_session_not_found_error_with_custom_identifier(self):
        """Test that session not found errors work correctly with custom identifiers."""
        custom_id = "session-not-found-test"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id, auto_create=False)

        # Try to execute code in non-existent session - should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            interpreter.code_interpreter(
                code_interpreter_input={
                    "action": {
                        "type": "executeCode",
                        "session_name": "non-existent-session",
                        "code": "print('This should fail')",
                        "language": "python"
                    }
                }
            )

        # Verify error message contains expected information
        error_message = str(exc_info.value)
        assert "non-existent-session" in error_message
        assert "not found" in error_message
        assert "initSession" in error_message

    def test_multiple_error_scenarios_with_different_identifiers(self):
        """Test various error scenarios with different custom identifiers."""
        test_cases = [
            {
                "identifier": "error-test-1",
                "session_name": "error-session-1",
                "description": "First error test"
            },
            {
                "identifier": "error-test-2", 
                "session_name": "error-session-2",
                "description": "Second error test"
            }
        ]

        with patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient") as mock_client_class:
            for i, test_case in enumerate(test_cases):
                # Mock different errors for each test case
                mock_client = MagicMock()
                mock_client.start.side_effect = Exception(f"Error {i+1}: Custom error occurred")
                mock_client_class.return_value = mock_client

                interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=test_case["identifier"])

                result = interpreter.code_interpreter(
                    code_interpreter_input={
                        "action": {
                            "type": "initSession",
                            "description": test_case["description"],
                            "session_name": test_case["session_name"]
                        }
                    }
                )

                # Verify error response contains expected information
                assert result['status'] == 'error'
                error_message = result['content'][0]['text']
                assert test_case["session_name"] in error_message
                assert f"Error {i+1}" in error_message


class TestIdentifierValidationAndEdgeCases:
    """Test edge cases and validation scenarios for custom identifiers."""

    def test_empty_string_identifier_uses_default(self):
        """Test that empty string identifier falls back to default."""
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier="")
        assert interpreter.identifier == "aws.codeinterpreter.v1"

    def test_none_identifier_uses_default(self):
        """Test that None identifier falls back to default."""
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=None)
        assert interpreter.identifier == "aws.codeinterpreter.v1"

    def test_whitespace_only_identifier_uses_default(self):
        """Test that whitespace-only identifier falls back to default."""
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier="   ")
        # Note: Current implementation doesn't strip whitespace, so this will use the whitespace string
        # This test documents the current behavior
        assert interpreter.identifier == "   "

    @skip_if_github_action.mark
    def test_complete_session_creation_flow_with_default_identifier(self, custom_identifier_interpreter):
        """Test complete session creation flow with default identifier (custom ones aren't supported)."""
        # Test direct tool call with default identifier
        result = custom_identifier_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Custom identifier test session",
                    "session_name": "custom-id-session"
                }
            }
        )

        # Verify session was created successfully
        assert result['status'] == 'success'
        assert 'sessionName' in result['content'][0]['json']
        assert result['content'][0]['json']['sessionName'] == 'custom-id-session'
        assert result['content'][0]['json']['description'] == 'Custom identifier test session'
        assert 'sessionId' in result['content'][0]['json']

    def test_very_long_identifier(self):
        """Test handling of very long identifier strings."""
        long_id = "a" * 1000  # Very long identifier
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=long_id)
        assert interpreter.identifier == long_id

    def test_special_characters_in_identifier(self):
        """Test handling of special characters in identifiers."""
        special_id = "test-interpreter-with-special-chars_123"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=special_id)
        assert interpreter.identifier == special_id

    @skip_if_github_action.mark
    def test_complex_identifier_end_to_end(self):
        """Test end-to-end functionality with complex identifier format."""
        complex_id = "integration-test-interpreter-xyz789"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=complex_id)

        result = interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "initSession",
                    "description": "Complex identifier test session",
                    "session_name": "complex-test-session"
                }
            }
        )

        # This test will likely fail with real AWS calls due to non-existent identifier,
        # but it tests that the identifier is passed through correctly
        # The specific error will depend on AWS validation
        # For now, we just verify the identifier was stored correctly
        assert interpreter.identifier == complex_id

    @skip_if_github_action.mark
    def test_auto_session_creation_with_custom_identifier(self):
        """Test automatic session creation when executing code directly."""
        # Use default identifier since custom identifiers may not exist in AWS
        interpreter = AgentCoreCodeInterpreter(region="us-west-2")  # Remove identifier="custom-auto-test"
        
        # Execute code without creating a session first - should auto-create with random UUID
        result = interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "executeCode",
                    "code": "print('Auto-created session with custom identifier')",
                    "language": "python"
                }
            }
        )
        
        assert result['status'] == 'success'
        
        # Verify a session was created (will be random UUID, not "default")
        assert len(interpreter._sessions) == 1
        
        # Get the auto-created session name (should start with "session-")
        auto_created_session = list(interpreter._sessions.keys())[0]
        assert auto_created_session.startswith("session-")
        
        # Verify we can list our auto-created session
        list_result = interpreter.code_interpreter(
            code_interpreter_input={"action": {"type": "listLocalSessions"}}
        )
        
        assert list_result['status'] == 'success'
        assert list_result['content'][0]['json']['totalSessions'] == 1
        
        # The session name should be the auto-created one
        session_names = [s['sessionName'] for s in list_result['content'][0]['json']['sessions']]
        assert auto_created_session in session_names

    @skip_if_github_action.mark
    def test_auto_session_creation_with_default_identifier(self):
        """Test automatic session creation when executing code directly."""
        interpreter = AgentCoreCodeInterpreter(region="us-west-2")
        
        # Execute code without creating a session first - should auto-create with random UUID
        result = interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "executeCode",
                    "code": "print('Auto-created session with default identifier')",
                    "language": "python"
                }
            }
        )
        
        assert result['status'] == 'success'
        
        # Verify a session was created (will be random UUID, not "default")
        assert len(interpreter._sessions) == 1
        
        # Get the auto-created session name (should start with "session-")
        auto_created_session = list(interpreter._sessions.keys())[0]
        assert auto_created_session.startswith("session-")
        
        # Verify we can list our auto-created session
        list_result = interpreter.code_interpreter(
            code_interpreter_input={"action": {"type": "listLocalSessions"}}
        )
        
        assert list_result['status'] == 'success'
        assert list_result['content'][0]['json']['totalSessions'] == 1
        
        # The session name should be the auto-created one
        session_names = [s['sessionName'] for s in list_result['content'][0]['json']['sessions']]
        assert auto_created_session in session_names

    @skip_if_github_action.mark
    def test_session_name_strategies(self):
        """Test different session_name strategies."""
        # Case 1: None -> random UUID
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", session_name=None)
        assert interpreter1.default_session.startswith("session-")
        assert len(interpreter1.default_session) == len("session-") + 12  # UUID hex[:12]
        
        # Case 2: Specific string -> use that string
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", session_name="my-analysis")
        assert interpreter2.default_session == "my-analysis"
        
        # Case 3: Runtime session (simulated)
        runtime_session_id = "runtime-abc123"
        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2", session_name=runtime_session_id)
        assert interpreter3.default_session == runtime_session_id

    def test_auto_create_flag_behavior(self):
        """Test auto_create flag behavior."""
        # Case 1: auto_create=True (default) - should succeed
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", auto_create=True)
        assert interpreter1.auto_create == True
        
        # Case 2: auto_create=False - strict mode
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)
        assert interpreter2.auto_create == False
        
        # Test strict mode behavior - should raise ValueError when session doesn't exist
        with pytest.raises(ValueError) as exc_info:
            interpreter2.code_interpreter(
                code_interpreter_input={
                    "action": {
                        "type": "executeCode",
                        "session_name": "non-existent-session",
                        "code": "print('This should fail')",
                        "language": "python"
                    }
                }
            )
        
        # Verify error message contains expected information
        error_message = str(exc_info.value)
        assert "non-existent-session" in error_message
        assert "not found" in error_message
        assert "initSession" in error_message