"""
Strands integration package for BedrockAgentCore Code Sandbox tools.

This package contains the Strands-specific implementations of the Bedrock AgentCore Code Interpreter
tools using the @tool decorator with Pydantic models and inheritance-based architecture.
"""

from .agent_core_code_interpreter import AgentCoreCodeInterpreter
from .code_interpreter import CodeInterpreter

__all__ = [
    # Base classes
    "CodeInterpreter",
    # Platform implementations
    "AgentCoreCodeInterpreter",
]
