# Programmatic Tool Calling Tool for Strands Agents

## Summary

This PR introduces a new **Programmatic Tool Calling** tool for Strands Agents, inspired by [Anthropic's Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/programmatic-tool-calling) feature. It enables agents to write Python code that invokes other tools as callable functions, reducing API round-trips and enabling complex orchestration logic.

## Motivation

Traditional tool calling requires multiple round-trips between the agent and tools, which can be inefficient for:
- Multi-step workflows with many dependent tool calls
- Loops that process data with tools
- Conditional logic based on tool results
- Data aggregation across multiple tool invocations

This tool addresses these challenges by allowing the agent to write Python code that orchestrates multiple tool calls in a single execution, similar to how Anthropic's Programmatic Tool Calling works.

## Key Features

### 1. Tools as Callable Functions
Tools are exposed through a `tools` namespace:
```python
result = tools.calculator(expression="2 + 2")
content = tools.file_read(path="data.txt", mode="view")
```

### 2. Complex Orchestration Support
Write loops, conditionals, and data processing:
```python
# Loop with tool calls
total = 0
for i in range(5):
    result = tools.calculator(expression=f"{i} * 10")
    total += int(result)

# Conditional logic
if condition:
    data = tools.file_read(path="config.json", mode="view")
```

### 3. Output Capture
Captures stdout/stderr from executed code:
```python
print(f"Processing result: {result}")  # Captured in output
```

### 4. Tool Call Tracking
Records all tool calls made during execution for transparency:
```
Tool calls made: 3
  1. calculator({'expression': '1 + 1'}) -> success
  2. calculator({'expression': '2 + 2'}) -> success
  3. file_read({'path': 'test.txt', 'mode': 'view'}) -> success
```

### 5. Security Features
- Code validation for potentially dangerous patterns
- User confirmation required (unless `BYPASS_TOOL_CONSENT=true`)
- Optional tool filtering via `allowed_tools` parameter
- Safe imports provided (json, re, datetime, math, etc.)

## Usage Example

```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator, file_read

agent = Agent(tools=[programmatic_tool_caller, calculator, file_read])

result = agent.tool.programmatic_tool_caller(
    code='''
    # Calculate sum of squares
    total = 0
    for i in range(1, 6):
        square = tools.calculator(expression=f"{i} ** 2")
        total += int(square)
        print(f"{i}² = {square}")
    print(f"Sum of squares: {total}")
    '''
)
```

## Implementation Details

### Architecture
- **ToolProxy class**: Exposes agent tools as callable methods with callback routing
- **OutputCapture class**: Captures stdout/stderr during code execution
- **_execute_tool function**: Routes tool calls through the agent's tool registry
- **_validate_code function**: Validates code for security concerns

### Integration with Strands
- Uses the `@tool` decorator for proper Strands integration
- Calls tools directly with keyword arguments (compatible with DecoratedFunctionTool)
- Handles both string and dict return values from tools
- Automatically excludes itself from available tools to prevent recursion

## Testing

The PR includes comprehensive unit tests covering:
- ToolProxy functionality (list, call, history, errors)
- Code validation (dangerous patterns detection)
- Tool execution (success, errors, result handling)
- Full integration tests with real tools
- Edge cases (empty code, only comments, exceptions)

**39 tests, all passing.**

## Checklist

- [x] Code follows the repo's coding standards
- [x] Conventional commit message format
- [x] Type annotations included
- [x] Google-style docstrings
- [x] Unit tests with good coverage
- [x] Integration tests included
- [x] Linting passes (ruff check)
- [x] Formatting passes (ruff format)

## Related Research

This implementation draws inspiration from:
- [Anthropic's Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/programmatic-tool-calling)
- [Advanced Tool Use announcement](https://www.anthropic.com/engineering/advanced-tool-use)
- Existing Strands tools patterns (python_repl, use_agent)
