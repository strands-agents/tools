"""
Strands integration package for BedrockAgentCore Code Sandbox tools.

This package contains the Strands-specific implementations of the Bedrock AgentCore Code Interpreter
tools using the @tool decorator with Pydantic models and inheritance-based architecture.

The AgentCoreCodeInterpreter class supports both default AWS code interpreter environments
and custom environments specified by identifier, allowing for flexible deployment
across different AWS accounts, regions, and custom code interpreter configurations.

Key Features:
    - Support for Python, JavaScript, and TypeScript code execution
    - Custom code interpreter identifier support
    - Session-based code execution with file operations
    - Full backward compatibility with existing implementations
    - Comprehensive error handling and logging

Example:
    >>> from strands_tools.code_interpreter import AgentCoreCodeInterpreter
    >>>
    >>> # Default usage
    >>> interpreter = AgentCoreCodeInterpreter(region="us-west-2")
    >>>
    >>> # Custom identifier usage
    >>> custom_interpreter = AgentCoreCodeInterpreter(
    ...     region="us-west-2",
    ...     identifier="my-custom-interpreter-abc123"
    ... )
"""

from .agent_core_code_interpreter import AgentCoreCodeInterpreter
from .code_interpreter import CodeInterpreter

__all__ = ["CodeInterpreter", "AgentCoreCodeInterpreter"]
