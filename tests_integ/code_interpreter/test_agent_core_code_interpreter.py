"""
Integration tests for the Bedrock AgentCore code interpreter platform.

These tests use the actual Bedrock AgentCore client to make real requests to the sandbox,
testing the complete end-to-end flow through the Agent interface.
"""

import logging

import pytest
from strands import Agent
from strands.agent import AgentResult
from strands_tools.code_interpreter import AgentCoreCodeInterpreter

from tests_integ.ci_environments import skip_if_github_action

logger = logging.getLogger(__name__)

@pytest.fixture
def bedrock_agent_core_code_interpreter() -> AgentCoreCodeInterpreter:
    """Create a real BedrockAgentCore code interpreter tool."""
    return AgentCoreCodeInterpreter(
        region="us-west-2",
        persist_sessions=False  # Don't persist for integration tests
    )


@pytest.fixture
def agent(bedrock_agent_core_code_interpreter: AgentCoreCodeInterpreter) -> Agent:
    """Create an agent with the BedrockAgentCore code interpreter tool."""
    return Agent(tools=[bedrock_agent_core_code_interpreter.code_interpreter])


@skip_if_github_action.mark
def test_direct_tool_call(agent):
    """Test code interpreter direct tool call."""

    result = agent.tool.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession",
                "description": "Data analysis session",
                "session_name": "analysissession"  # Cleaned name (no dashes)
            }
        }
    )

    assert result['status'] == 'success'


@skip_if_github_action.mark
def test_complex_natural_language_workflow(agent):
    """Test complex multistep workflow through natural language instructions."""
    
    result: AgentResult = agent("""
You have code execution capabilities. Complete this workflow efficiently by combining operations where possible:

1. Create a Python script 'data_generator.py' that generates a CSV file 'sales_data.csv' with 20 rows (date, product, quantity, price, customer_id columns) and executes it

2. Verify the CSV was created by reading 'sales_data.csv'

3. Create and run a shell script 'cleanup.sh' that lists all files

After ALL steps complete successfully, respond with ONLY the word "PASS". If any step fails, respond with "FAIL".
    """)

    assert "PASS" in result.message["content"][0]["text"]

@skip_if_github_action.mark
def test_auto_session_creation(bedrock_agent_core_code_interpreter):
    """Test automatic session creation on first code execution."""
    # Execute code directly without initializing session
    result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "code": "import platform\nprint(f'Running on Python {platform.python_version()}')",
                "language": "python"
            }
        }
    )
    
    assert result['status'] == 'success'
    
    # Verify a session was auto-created (will be random UUID, not "default")
    assert len(bedrock_agent_core_code_interpreter._sessions) == 1
    auto_created_session = list(bedrock_agent_core_code_interpreter._sessions.keys())[0]
    assert auto_created_session.startswith("session")  # Check pattern instead of exact name
    
    # Execute a second command in the same auto-created session
    result2 = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "code": "print('Second execution in auto-created session')",
                "language": "python"
            }
        }
    )
    
    assert result2['status'] == 'success'
    
    # Verify still only one session exists (reused the auto-created one)
    assert len(bedrock_agent_core_code_interpreter._sessions) == 1