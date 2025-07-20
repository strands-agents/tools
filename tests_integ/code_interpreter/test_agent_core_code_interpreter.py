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
    return AgentCoreCodeInterpreter(region="us-west-2")


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
                "session_name": "analysis-session"
            }
        }
    )

    assert result['status'] == 'success'


@skip_if_github_action.mark
def test_complex_natural_language_workflow(agent):
    """Test complex multistep workflow through natural language instructions."""
    # Complex natural language instruction that requires multiple operations
    result: AgentResult = agent("""
    I need you to help me with a complex data processing task. Please:
    
    1. Create a new code session for this data processing project
    
    2. Write a Python script called 'data_generator.py' that:
       - Creates a CSV file called 'sales_data.csv' with sample sales data
       - Include columns: date, product, quantity, price, customer_id
       - Generate at least 20 rows of realistic sample data
       - Save this data to the CSV file

    3. Write another Javascript script called 'data_processor.js' that:
       - Reads the 'sales_data.csv' file you just created
       - Creates a summary report and saves it to 'sales_report.txt'
    
    4. Run both scripts in sequence
    
    5. Read the contents of both the CSV file and the report file to verify they were created correctly
    
    6. Finally, create a shell script called 'cleanup.sh' that lists all the files we created, then run it
    
    7. If and only if all steps succeed RESPOND WITH ONLY "PASS" IF NOT RETURN "FAIL"
    """)

    assert "PASS" in result.message["content"][0]["text"]