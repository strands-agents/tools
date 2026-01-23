## Description

This PR adds a new `skills` tool that implements support for [Agent Skills](https://agentskills.io) - modular packages of specialized instructions that help AI agents perform specific tasks effectively.

### What are Agent Skills?

Agent Skills are folders containing `SKILL.md` files with structured instructions. They follow the [AgentSkills.io specification](https://agentskills.io) and enable:

- **Progressive Disclosure**: Load only skill metadata initially (~100 tokens), full instructions on demand
- **Modular Knowledge**: Package domain-specific expertise as reusable skills
- **No Code Changes**: Add/remove capabilities via markdown files

### Tool Design

The tool follows the action-based pattern used by other tools (like `mcp_client`, `memory`, `slack`):

```python
from strands import Agent
from strands_tools import skills

agent = Agent(tools=[skills])

# List available skills (auto-discovered from STRANDS_SKILLS_DIR)
agent.tool.skills(action="list")

# Use a skill - loads its instructions into conversation
agent.tool.skills(action="use", skill_name="code-reviewer")

# Load a resource file from a skill
agent.tool.skills(action="get_resource", skill_name="code-reviewer", resource_path="scripts/analyze.py")
```

### Actions

| Action | Description |
|--------|-------------|
| `list` | Show all available skills with descriptions |
| `use` | Load a skill's full instructions (returned as tool result) |
| `get_resource` | Load a script/reference/asset file from a skill |
| `list_resources` | List all files available in a skill |

### Design Decisions

1. **Simple stateless design**: Skills are loaded on demand and returned as tool results. No state tracking or system prompt modification - this matches how [Anthropic's skills](https://github.com/anthropics/skills) and [AWS sample implementation](https://github.com/aws-samples/sample-strands-agents-agentskills) work.

2. **No activate/deactivate**: These concepts don't exist in the AgentSkills.io spec. Skills are simply loaded when needed via conversation history.

3. **Auto-discovery**: Skills are automatically discovered from `STRANDS_SKILLS_DIR` (default: `./skills`) on first use.

4. **SKILL.md format**: Follows AgentSkills.io spec with YAML frontmatter for metadata and markdown body for instructions.

### Skill Directory Structure

```
skills/
├── code-reviewer/
│   ├── SKILL.md              # Required: YAML frontmatter + instructions
│   ├── scripts/              # Optional: Executable code
│   │   └── analyze.py
│   └── references/           # Optional: Additional documentation
│       └── security.md
└── data-analyst/
    └── SKILL.md
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRANDS_SKILLS_DIR` | `./skills` | Directory containing skill folders |

## Related Issues

- strands-agents/sdk-python#1181

## Documentation PR

N/A - Documentation is included in README.md updates in this PR.

## Type of Change

New Tool

## Testing

- [x] 13 unit tests covering all actions, error handling, and auto-discovery
- [x] All tests pass (`python -m pytest tests/test_skills.py -v`)
- [x] Linting passes (`ruff check src/strands_tools/skills.py tests/test_skills.py`)

Test coverage includes:
- Skill listing (empty, non-existent, valid directories)
- Skill usage (valid, non-existent, missing parameters)
- Resource handling (list, get, path traversal protection)
- Error handling (invalid actions, missing required params)
- Auto-discovery from environment variable

## Checklist

- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [x] I have updated the documentation accordingly
- [x] I have added an appropriate example to the documentation to outline the feature
- [x] My changes generate no new warnings
- [x] Any dependent changes have been merged and published

---

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
