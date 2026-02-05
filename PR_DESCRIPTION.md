## Description

Add `programmatic_tool_caller` tool that enables programmatic/code-based tool invocation, similar to Anthropic's Programmatic Tool Calling feature. This allows an agent to write Python code that calls other tools as functions, reducing API round-trips and enabling complex orchestration logic.

### Key Features

- **Tools exposed as async functions**: `await calculator(expression="2+2")`
- **Sync versions also available**: `calculator_sync(expression="2+2")`
- **Only `print()` output returned**: Matches Anthropic's design - tool results stay in code execution context, don't enter agent's context window unless explicitly printed
- **Complex orchestration**: Loops, conditionals, data filtering all work naturally in Python

### Example Usage

```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator

agent = Agent(tools=[programmatic_tool_caller, calculator])

result = agent.tool.programmatic_tool_caller(
    code="""
import asyncio

async def main():
    # Calculate sum of squares 1-5
    total = 0
    for i in range(1, 6):
        square = await calculator(expression=f"{i} ** 2")
        total += int(square)
    print(f"Sum of squares: {total}")  # Only this goes to agent

asyncio.run(main())
"""
)
```

### Alignment with Anthropic's Design

| Aspect | Anthropic | Our Implementation |
|--------|-----------|-------------------|
| Tool Syntax | `await tool(...)` | `await tool(...)` ✓ |
| Output | Only `print()` | Only `print()` ✓ |
| Tool Messages | Not in conversation | Not in conversation ✓ |
| Execution | Sandboxed container | Python `exec()` |

### Not Implemented (intentionally)

- `allowed_callers` configuration - all tools available by default
- Container management - we use local exec
- `caller` field tracking in responses

## Related Issues

N/A - New feature

## Type of Change

New Tool

## Testing

- [x] 51 unit and integration tests pass
- [x] Linting passes (ruff check, ruff format)
- [x] Tests cover async/sync execution, loops, conditionals, data filtering, error handling

## Checklist

- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [x] I have updated the documentation accordingly (README.md)
- [x] I have added an appropriate example to the documentation to outline the feature
- [x] My changes generate no new warnings
