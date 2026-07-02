# Dakera Memory Tool

The `dakera_memory` tool provides persistent, decay-weighted agent memory for Strands agents, backed by a self-hosted [Dakera](https://github.com/dakera-ai/dakera-deploy) memory server.

Unlike key-value stores, Dakera ranks recalled memories by **importance**, **recency**, and **semantic relevance** — so the most useful context surfaces first.

## Prerequisites

### 1. Run a Dakera server

Use the official Docker Compose stack (includes the Dakera API server and MinIO for object storage):

```bash
git clone https://github.com/dakera-ai/dakera-deploy
cd dakera-deploy
docker compose up -d
```

The API server starts on **port 3000** by default.

### 2. Install the optional dependency

```bash
pip install 'strands-agents-tools[dakera-memory]'
```

### 3. Configure environment variables

| Variable | Default | Description |
|---|---|---|
| `DAKERA_BASE_URL` | `http://localhost:3000` | Dakera server URL |
| `DAKERA_API_KEY` | _(none)_ | API key (`dk-...`) if authentication is enabled |

## Usage

```python
from strands import Agent
from strands_tools import dakera_memory

agent = Agent(tools=[dakera_memory])

# Store a memory with importance weighting
agent.tool.dakera_memory(
    action="store",
    agent_id="alex",
    content="Alex prefers concise bullet-point answers over long prose.",
    importance=0.85,
    memory_type="semantic",
    metadata={"category": "preferences"},
)

# Semantic recall — decay-weighted, returns most relevant memories first
agent.tool.dakera_memory(
    action="retrieve",
    agent_id="alex",
    query="user communication style",
    top_k=5,
)

# Fetch a specific memory by ID
agent.tool.dakera_memory(
    action="get",
    agent_id="alex",
    memory_id="mem_abc123",
)

# Update a memory's content
agent.tool.dakera_memory(
    action="update",
    agent_id="alex",
    memory_id="mem_abc123",
    content="Alex prefers bullet points and code examples.",
)

# Delete a memory
agent.tool.dakera_memory(
    action="delete",
    agent_id="alex",
    memory_id="mem_abc123",
)
```

## Actions

| Action | Required params | Optional params | Description |
|---|---|---|---|
| `store` | `agent_id`, `content` | `importance`, `memory_type`, `metadata` | Store a new memory |
| `retrieve` | `agent_id`, `query` | `top_k` (default 5) | Decay-weighted semantic search |
| `get` | `agent_id`, `memory_id` | — | Fetch one memory by ID |
| `update` | `agent_id`, `memory_id` | `content`, `metadata`, `memory_type` | Update an existing memory |
| `delete` | `agent_id`, `memory_id` | — | Remove a memory |

### Memory types

| Type | Use case |
|---|---|
| `episodic` | Specific events or interactions (default) |
| `semantic` | Facts, preferences, general knowledge |
| `procedural` | Workflows and how-to knowledge |
| `working` | Short-lived context for the current session |

### Importance scoring

`importance` is a float from `0.0` to `1.0`. Higher values make a memory harder to decay and more likely to surface in recall:

- `0.9–1.0` — critical facts (e.g. user identity, strong preferences)
- `0.7–0.8` — important but routine (e.g. frequently-used configs)
- `0.4–0.6` — background context

## Safety

Mutative operations (`store`, `update`, `delete`) display a confirmation panel before executing. Set `BYPASS_TOOL_CONSENT=true` to skip confirmations in automated/test environments.

## Integration tests

Integration tests require a live Dakera server (see Prerequisites). Run them with:

```bash
DAKERA_BASE_URL=http://localhost:3000 pytest tests/test_dakera_memory.py -v
```

Unit tests (no server needed) run with:

```bash
BYPASS_TOOL_CONSENT=true pytest tests/test_dakera_memory.py -v
```
