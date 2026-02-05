## Description

Add `programmatic_tool_caller` tool that enables programmatic/code-based tool invocation, similar to Anthropic's Programmatic Tool Calling feature.

### Key Features

- **Tools as async functions**: `await calculator(expression="2+2")`
- **Auto async context**: No `asyncio.run()` boilerplate - code runs in async context automatically
- **Parallel execution**: Use `asyncio.gather()` for concurrent tool calls
- **Only print() returned**: Tool results stay in code context unless printed

### Example

```python
result = agent.tool.programmatic_tool_caller(
    code="""
# Simple
result = await calculator(expression="2 + 2")
print(result)

# Parallel
results = await asyncio.gather(
    calculator(expression="1+1"),
    calculator(expression="2+2"),
)
print(results)
"""
)
```

## Type of Change

New Tool

## Testing

- [x] 29 tests pass (unit + integration)
- [x] Linting passes

## Checklist

- [x] Tests added
- [x] README.md updated
- [x] No new warnings
