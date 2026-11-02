# Deprecations

The tools below are deprecated, and each row points to where that capability now lives.

When this package started, the SDK had no built-in way to do most of these things, so we shipped
tools to fill the gap. Much of that is now native: the SDK reasons, injects context, manages memory,
and runs tools concurrently on its own, and where a vendor maintains an official MCP server, that
server will always track their API better than a wrapper here can. Keeping a second implementation
alongside means two things to fix and two things to drift, so we would rather point you at the one
that gets the attention.

More tools will follow as their capabilities land elsewhere, and this repository will eventually be
archived. Nothing breaks suddenly — but migrating when a tool is first deprecated is easier than
moving several at once later.

Deprecated tools keep working until v0.9.0. Each one logs a warning when invoked, starting in the
release shown below, and in **v0.9.0** that warning becomes an import error, so the tool can no longer be imported.

They are also marked with `@typing_extensions.deprecated`, so type checkers and IDEs flag
usage before you run anything. To list what you still need to migrate, run
`mypy --enable-error-code deprecated` over your project: it reports the import line for each
deprecated tool, without invoking any of them.
Prefer mypy here: pyright, with `reportDeprecated` enabled, reports direct calls such as
`calculator(expression=...)` but not `agent.tool.calculator(...)`, and it reaches the import
line for only the tools whose marker is not wrapped by `@tool`. At runtime, Python suppresses the
`DeprecationWarning` that function tools raise when the agent invokes them, and the markers on
class-based tools are static-only, which is why the log message exists as well.

| Tool | Suggested alternative | Warning | Error |
|------|-----------------------|---------|-------|
| `sleep` | `from strands.vended_tools import sleep` | v0.8.6 | v0.9.0 |
| `editor` | `from strands.vended_tools import edit, read, write` | v0.8.6 | v0.9.0 |
| `shell` | `from strands.vended_tools import shell` | v0.8.6 | v0.9.0 |
| `batch` | none needed — concurrent tool execution is the SDK default ([docs](https://strandsagents.com/docs/user-guide/concepts/tools/executors/)) | v0.8.6 | v0.9.0 |
| `think` | native extended thinking via model reasoning config ([docs](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/)) | v0.8.6 | v0.9.0 |
| `current_time` | `ContextInjector` ([docs](https://strandsagents.com/docs/user-guide/concepts/plugins/context-injector/)) | v0.8.6 | v0.9.0 |
| `memory` | `MemoryManager` + `BedrockKnowledgeBaseStore` ([docs](https://strandsagents.com/docs/user-guide/concepts/memory/bedrock-knowledge-base/)) | v0.8.6 | v0.9.0 |
| `retrieve` | `MemoryManager` + `BedrockKnowledgeBaseStore(writable=False)` ([docs](https://strandsagents.com/docs/user-guide/concepts/memory/overview/)) | v0.8.6 | v0.9.0 |
| `calculator` | `from strands.vended_tools import shell` (run `python3 -c` with sympy) | v0.8.6 | v0.9.0 |
| `cron` | `from strands.vended_tools import shell` (manage `crontab`), or Amazon EventBridge Scheduler | v0.8.6 | v0.9.0 |
| `environment` | `from strands.vended_tools import shell` (inspect only, see notes) | v0.8.6 | v0.9.0 |
| `bright_data` | [Bright Data's official MCP server](https://docs.brightdata.com/ai/mcp-server/overview) | v0.8.9 | v0.9.0 |
| `http_request` | `from strands.vended_tools import http_request, web_fetch` (http_request for APIs, web_fetch for web pages)| v0.8.9 | v0.9.0 |
| `slack` | [official Slack MCP server](https://docs.slack.dev/ai/mcp-server/); `slack_bolt` for Socket Mode | v0.8.6 | v0.9.0 |
| `diagram` | no replacement — have the model write graphviz/mermaid/`diagrams` code directly | v0.8.6 | v0.9.0 |
| `rss` | no replacement — parse feeds directly with `feedparser` | v0.8.6 | v0.9.0 |
| `tavily` (`tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`) | [official Tavily MCP server](https://docs.tavily.com/documentation/mcp) | v0.8.9 | v0.9.0 |
| `exa` (`exa_search`, `exa_get_contents`) | [official Exa MCP server](https://exa.ai/docs/reference/exa-mcp) | v0.8.9 | v0.9.0 |
| `search_video` | [official TwelveLabs MCP servers](https://docs.twelvelabs.io/docs/advanced/model-context-protocol) | v0.8.9 | v0.9.0 |
| `chat_video` | [official TwelveLabs MCP servers](https://docs.twelvelabs.io/docs/advanced/model-context-protocol) | v0.8.9 | v0.9.0 |
| `journal` | `from strands.vended_tools import notebook` | v0.8.9 | v0.9.0 |
| `python_repl` | `from strands.vended_tools import python_repl` | v0.8.10 | v0.9.0 |
| `a2a_client` | `from strands.vended_tools import make_a2a_client` | v0.8.10 | v0.9.0 |
| `mcp_client` | `from strands.vended_tools import make_mcp_router` | v0.8.10 | v0.9.0 |
| `mem0_memory` (`mem0_memory`, `Mem0MemoryTool`) | `MemoryManager` + `Mem0MemoryStore` from [mem0-strands](https://docs.mem0.ai/integrations/strands) | v0.8.10 | v0.9.0 |
| `code_interpreter` (`AgentCoreCodeInterpreter`) | `from strands.vended_tools import python_repl` for Python; `from strands.vended_tools import shell` for other languages | v0.8.11 | v0.9.0 |
| `handoff_to_user` | `from strands.vended_tools import handoff_to_user` | v0.8.11 | v0.9.0 |
| `speak` | no replacement — call Amazon Polly's `synthesize_speech` directly | v0.8.11 | v0.9.0 |
| `load_tool` | no replacement — pass tools to the `Agent` directly, or use `Agent(load_tools_from_directory=True)` | v0.8.11 | v0.9.0 |
| `use_computer` | no replacement — drive `pyautogui` and `pytesseract` directly | v0.8.11 | v0.9.0 |
| `browser` (`LocalChromiumBrowser`, `AgentCoreBrowser`) | no replacement — drive Playwright directly, with `bedrock-agentcore`'s `BrowserClient` for AgentCore sessions | v0.8.11 | v0.9.0 |
| `nova_reels` | no replacement — Amazon Nova Reel reached end-of-life on Bedrock ([docs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html)) | v0.8.11 | v0.9.0 |
| `agent_core_memory` (`AgentCoreMemoryToolProvider`) | `MemoryManager` + `AgentCoreMemoryStore` from `bedrock-agentcore` ([docs](https://strandsagents.com/docs/integrations/memory-stores/agentcore-memory-store/)) | v0.8.11 | v0.9.0 |
| `generate_image_stability` | no replacement — call the Stability AI REST API directly ([docs](https://docs.stability.ai/api-reference)) | v0.8.11 | v0.9.0 |
| `generate_image` | no replacement — call Bedrock's `invoke_model` with a Stability AI model directly ([docs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html)) | v0.8.11 | v0.9.0 |
| `image_reader` | image content blocks in the agent prompt ([docs](https://strandsagents.com/docs/user-guide/sdk/agents/prompts/)) | v0.8.11 | v0.9.0 |
| `elasticsearch_memory` (`elasticsearch_memory`, `ElasticsearchMemoryTool`) | no replacement — implement a `MemoryStore` over [`elasticsearch-py`](https://www.elastic.co/docs/reference/elasticsearch/clients/python) for `MemoryManager` ([docs](https://strandsagents.com/docs/user-guide/sdk/memory/managing-memory/#custom-stores)) | v0.8.11 | v0.9.0 |
| `swarm` | `from strands.vended_tools import swarm` | v0.8.11 | v0.9.0 |
| `use_agent` | `from strands.vended_tools import subagent` | v0.8.11 | v0.9.0 |
| `use_llm` | `from strands.vended_tools import subagent` | v0.8.11 | v0.9.0 |
| `file_read` | `from strands.vended_tools import read` | v0.8.12 | v0.9.0 |
| `file_write` | `from strands.vended_tools import write` | v0.8.12 | v0.9.0 |
| `graph` | `from strands.vended_tools import graph` | v0.8.12 | v0.9.0 |
| `workflow` | `from strands.vended_tools import graph` | v0.8.12 | v0.9.0 |
| `agent_graph` | `from strands.vended_tools import graph` | v0.8.12 | v0.9.0 |
| `stop` | `from strands.vended_tools import stop` | v0.8.12 | v0.9.0 |
| `use_aws` | [AWS MCP Server](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/mcp-server.html) | v0.8.12 | v0.9.0 |
| `mongodb_memory` (`mongodb_memory`, `MongoDBMemoryTool`) | no replacement — implement a `MemoryStore` over [`pymongo`](https://pymongo.readthedocs.io/) for `MemoryManager` ([docs](https://strandsagents.com/docs/user-guide/sdk/memory/managing-memory/#custom-stores)) | v0.8.12 | v0.9.0 |

```python
# Before
from strands import Agent
from strands_tools import editor, shell, sleep

agent = Agent(tools=[editor, shell, sleep])

# After
from strands import Agent
from strands.vended_tools import edit, read, shell, sleep, write

agent = Agent(tools=[edit, read, shell, sleep, write])
```

## Behavior differences

The replacements are not drop-in equivalents. Check these before migrating:

- **`strands_tools.shell` → `strands.vended_tools.shell`**: the SDK tool routes commands through the agent's configured
  sandbox and is stateless — each call runs in a fresh shell, so variables and the working
  directory do not persist between calls. It does not provide PTY support or batched
  parallel/sequential command execution.
- **`editor` → `read` / `write` / `edit`**: `view` maps to `read`, `create` to `write`, and `str_replace` to
  `edit`. `edit` requires the old string to be unique in the file, whereas `str_replace` replaced every occurrence.
  `write` and `edit` write immediately, with no confirmation prompt and no `.bak` backup. The `insert`,
  `pattern_replace`, `find_line`, and `undo_edit` commands have no SDK equivalent. Directory views move to `shell`.
- **`sleep` → `sleep`**: the maximum duration is set with
  `make_sleep(max_duration=...)` and defaults to 60 seconds, replacing the
  `MAX_SLEEP_SECONDS` environment variable which defaulted to 300 seconds.
- **`batch`** — you can drop it. The SDK runs tool calls concurrently by default via
  `ConcurrentToolExecutor`, which also keeps tracing, metrics, and hooks intact. See
  [Tool executors](https://strandsagents.com/docs/user-guide/concepts/tools/executors/).
- **`think` → extended thinking** — reasoning is now a model capability rather than a tool, so you
  configure it once instead of prompting for it. It is single-pass, so if you were relying on
  `cycle_count` to refine across passes, or on tool calls inside the reasoning loop, that pattern
  moves into your own orchestration. Config is per provider — see
  [Amazon Bedrock](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/).
- **`current_time` → `ContextInjector`** — the date is context, not something the agent should have to
  ask for, so injecting it means it is always fresh and the model cannot forget to call it. The
  trade-off is that your code picks the timezone; if you need the model to choose one per call, the
  official [MCP `time` server](https://github.com/modelcontextprotocol/servers/tree/main/src/time)
  still does that. See
  [ContextInjector](https://strandsagents.com/docs/user-guide/concepts/plugins/context-injector/).
- **`memory` / `retrieve` → memory stores** — memory is a first-class SDK concept now, so stores plug
  into the agent and inject context automatically instead of waiting to be called. `store`/`retrieve`
  become `add`/`search`. `list`, `get`, and `delete` are not part of the store protocol, so keep them
  as backend-native tools via `get_tools()`. One thing to know: the store returns relevance scores but
  does not filter on them, so if you relied on `retrieve`'s default `score >= 0.4` floor, apply it
  yourself. See
  [Bedrock Knowledge Base store](https://strandsagents.com/docs/user-guide/concepts/memory/bedrock-knowledge-base/)
  and [Custom stores](https://strandsagents.com/docs/user-guide/concepts/memory/overview/#custom-stores).
- **`calculator` → `shell`** — run `python3 -c` with sympy for the same symbolic math. Worth being
  explicit: `shell` runs whatever it is given, whereas this tool checked expressions against an
  allowlist first. If you are evaluating untrusted input, put a sandbox behind the shell tool or keep
  your own validation in front of it.
- **`cron` → `shell`** — `crontab` through the shell tool covers scheduling on a host. Two things to
  plan for: the shell is sandbox-routed, so under Docker or SSH your jobs land in an environment that
  may not have a cron daemon, and you will be composing crontab lines yourself rather than using
  structured `list`/`add`/`remove`/`edit` actions. For scheduling that outlives the host, Amazon EventBridge
  Scheduler is usually the better fit.
- **`environment` → `shell`** — `env` and `printenv` cover reading. Setting is genuinely different: a
  child shell cannot change the agent's own process environment, so variables need to be set where the
  agent is launched, or passed per call. If you were leaning on `PROTECTED_VARS` or secret masking,
  that guarding moves to your side.
- **`bright_data` → Bright Data's official MCP server** — the vendor-maintained server covers search,
  scraping, structured extraction, and browser automation. Its tool names and response shapes are not
  drop-in equivalents, so follow Bright Data's
  [setup and migration documentation](https://docs.brightdata.com/ai/mcp-server/overview).
- **`http_request` → `http_request` / `web_fetch`** — the tool splits by intent: raw API calls move to
  the vended `http_request`, while fetching and reading a page moves to `web_fetch`. Authentication and
  endpoint scoping now live in an `httpx.AsyncClient` you pass via `make_http_request(client=...)`.
  `web_fetch` defaults to an agentic mode that summarizes a page before it enters the main context;
  pass `mode="markdown"` to the `web_fetch` factory for the full page as markdown, replacing
  `convert_to_markdown=True`.
- **`slack` → Slack's official MCP server** — Slack maintains it, so it tracks their API directly, and
  it uses OAuth rather than long-lived tokens. It exposes a curated tool set rather than this tool's
  passthrough to any Web API method, and being request/response it does not cover Socket Mode or
  real-time events — `slack_bolt` remains the right tool for event listeners. Endpoint and setup:
  [docs.slack.dev/ai/mcp-server](https://docs.slack.dev/ai/mcp-server/).
- **`diagram`** — no direct replacement, and that is deliberate: this tool was a wrapper over
  `graphviz` and the `diagrams` package, and a capable model writes that code well on its own. The
  wrapper mostly added a layer to keep in sync.
- **`rss`** — no direct replacement. Fetching and parsing a feed is a few lines of `feedparser`, and
  the subscription list was a JSON file. If you were using feed HTTP Basic auth or the stored
  subscriptions, those move into your code.
- **`tavily` → Tavily MCP server** — Auth moves from `TAVILY_API_KEY` to OAuth, a key in the server URL, or a header. A handful of parameters have no MCP equivalent:
  `auto_parameters`, `chunks_per_source`, `days`, `include_answer`, `categories`, `exclude_paths`,
  and the `"markdown"/"text"` literals for `include_raw_content` (the MCP accepts a boolean only). If you need these parameters, call the [Tavily REST API](https://docs.tavily.com/documentation/api-reference/introduction) directly.
- **`exa` → Exa MCP server** — the MCP renames the tools: `exa_search` becomes `web_search_exa` and
  `exa_get_contents` becomes `web_fetch_exa`. For the full parameter surface of `exa_search`, opt in
  to `web_search_advanced_exa` via the `tools` URL parameter. Auth moves from `EXA_API_KEY` to OAuth, a key in
  the server URL, or a header.
- **`search_video` / `chat_video` → TwelveLabs MCP servers** — Jockey adds cross-video reasoning, while single-video search and analysis continue
  at `mcp-alpic.twelvelabs.io`. Jockey requires publicly accessible URLs and has no equivalent for `chat_video`'s local-file upload nor
  `search_video`'s `group_by`, `threshold`, and `search_options` parameters.
- **`journal` → `notebook`** — `notebook` keeps notes in `agent.state` instead of `./journal/<date>.md`, so persistence follows your session
  manager. There is no `add_task` equivalent and no per-day entry, so tasks and dated entries become plain note text. If you need the entries as
  files on disk, use `write`.
- **`python_repl` → `python_repl`** — the SDK tool runs each call in a fresh interpreter, so in-memory state does not
  persist; save intermediate results to files instead. By default it executes on the host with no isolation and no
  confirmation prompt — pass a `sandbox=` to `Agent` (for example `DockerSandbox`) if you relied on the consent gate.
  There is no PTY, stdin, or `reset_state`, and the `PYTHON_REPL_*` environment variables no longer apply.
- **`a2a_client` → `make_a2a_client`** — the three `a2a_*` tools become one tool with `discover` and
  `send_message` operations, limited to the endpoints you pass in `allowed_endpoints`. Authentication,
  timeouts, and push notifications move from `httpx_client_args` and the webhook arguments into a
  per-endpoint `ClientConfig`.
- **`mcp_client` → `make_mcp_router`** — the model connects by name to servers you allowlist in `servers`
  rather than supplying its own command or URL. There is no `load_tools`, so remote tools are invoked
  through `call_tool`.
- **`mem0_memory` → `Mem0MemoryStore`** — plugs into `MemoryManager`, so memories are recalled automatically.
  `retrieve` becomes `search_memory` and `store` becomes `add_memory` (with `add_tool_config=True`); for `get`, `list`,
  `delete`, and `history`, use `mem0`'s `MemoryClient`. `mem0-strands` needs `mem0ai` 2.x, so drop the `mem0-memory`
  extra first. Backend env vars don't carry over: without `MEM0_API_KEY`, pass a Mem0 OSS `config=`. See the
  [Mem0 integration docs](https://docs.mem0.ai/integrations/strands).
- **`code_interpreter` → `python_repl` / `shell`** — both run in the agent's sandbox instead of an AgentCore
  session. An `Agent` with no `sandbox=` falls back to `NotASandboxLocalEnvironment`, which runs on the host
  with no isolation, so pass a sandbox (e.g. `strands.sandbox.docker.DockerSandbox`) if you were relying on
  AgentCore's. Each call starts a fresh process, so in-memory state does not persist. There are no named
  sessions or file actions; use `read`, `write`, `edit`, and/or `shell` for file operations.
- **`handoff_to_user` → `handoff_to_user`** — the SDK tool raises an interrupt instead of prompting on the
  console, so the agent stops with `stop_reason == "interrupt"` and your code resumes it with the user's
  reply. There is no `breakout_of_loop`: every call pauses the agent.
- **`speak`** — no direct replacement. Text-to-speech is a single Polly `synthesize_speech` call (or
  macOS `say`), so a custom tool covers it in a few lines. If you want the agent itself to talk,
  `from strands.bidi import BidiAgent` runs a live voice session with a speech model, though it does not
  read text verbatim or save audio files.
- **`load_tool`** — no direct replacement. Letting the model load arbitrary files as tools is rarely what
  you want; pass tools to the `Agent` up front, or set `load_tools_from_directory=True` to load and
  hot-reload tools from `./tools/`.
- **`use_computer`** — no direct replacement. This tool was a wrapper over `pyautogui`, `pytesseract`, and
  OS-specific app control; if you need desktop automation, call those from a custom tool scoped to the
  actions you actually use.
- **`browser`** — no direct replacement. This tool was a wrapper over Playwright, and `bedrock-agentcore`
  ships only the session layer (`BrowserClient`), not a Strands tool. Write a custom tool that drives
  Playwright directly; for AgentCore, start a session with `BrowserClient` and connect Playwright to it.
- **`nova_reels`** — no direct replacement. Amazon Nova Reel reached end-of-life on Bedrock on September 30, 2026,
  so this tool no longer works.
- **`agent_core_memory` → `AgentCoreMemoryStore`** — plugs into `MemoryManager`, so memories are recalled
  automatically. `retrieve` becomes `search_memory`; there is no `record` tool, so set `writable=True,
  extraction=True` and AgentCore extracts memories from the conversation. For `list`, `get`, and `delete`, use
  `bedrock-agentcore`'s `MemorySessionManager`.
- **`generate_image_stability`** — no direct replacement. This tool was a wrapper over Stability AI's
  [`v2beta/stable-image` REST API](https://docs.stability.ai/api-reference), so a custom tool covers it in a
  few lines.
- **`generate_image`** — no direct replacement. This tool was a wrapper over Bedrock's `invoke_model` for
  [Stability AI models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html), so a custom tool covers it in a few lines.
- **`image_reader` → image content blocks** — pass the image in the prompt, e.g.
  `agent([{"text": "..."}, {"image": {"format": "png", "source": {"bytes": data}}}])`. The model can no longer load
  images by path mid-run; if it needs to, return that same image block from a custom tool.
- **`elasticsearch_memory` → custom `MemoryStore`** — no store ships for Elasticsearch, but one only needs `search`
  (a k-NN query) and `add` (index a document) over
  [`elasticsearch-py`](https://www.elastic.co/docs/reference/elasticsearch/clients/python). `retrieve` then becomes
  `search_memory` and `record` becomes `add_memory` (with `writable=True` on the store and `add_tool_config=True` on
  `MemoryManager`); bind the namespace in the store, not the tool call, to keep tenants isolated. For `list`, `get`, and `delete`, call `elasticsearch-py` directly.
- **`swarm` → `swarm`** — by default every agent inherits the
  parent's model and all of its tools, and the model writes only `instructions` (formerly `system_prompt`). To let
  it pick tools or models per agent again, offer a set with `make_swarm(tools=Choice([...], multiple=True),
  model=Choice([...]))`. The handoff, iteration, and timeout limits also move from tool arguments to `make_swarm(...)`.
- **`use_agent` / `use_llm` → `subagent`** — by default the child inherits the parent's model and tools, and the
  model passes a `task` with optional `instructions` (without them, a built-in general-purpose prompt is used). In place
  of `use_agent`'s free-form `model_provider` and `model_settings`, offer models with `make_subagent(model=Choice([...]))`
  or pin one with `Fixed(...)`. The result is the child's final message, without the metrics block.
  `Choice`, `Fixed`, and the other axis modes come from `strands.multiagent.spec`.
- **`file_read` → `read`** — the SDK tool reads one file per call by absolute path, returning numbered lines paged
  with `offset` and `limit`; images and binary documents come back as content blocks, while the csv, html, txt, and md
  files that `document` mode returned as document blocks now come back as numbered text. The `find`, `search`, `stats`,
  `preview`, `chunk`, `diff`, and `time_machine` modes have no equivalent, so run those through `shell`.
- **`file_write` → `write`** — the SDK tool writes through the agent's sandbox and takes absolute paths only. It does
  not ask for confirmation or show a preview, so `BYPASS_TOOL_CONSENT` no longer applies. If you relied on the consent
  prompt, raise an interrupt from a `BeforeToolCallEvent` hook to approve each write. See
  [Interrupts](https://strandsagents.com/docs/user-guide/sdk/interrupts/#hooks).
- **`graph` → `graph`** — the model passes `task`, `nodes`, and `edges` in one call and gets each node's output back,
  with no create/execute/status/list/delete lifecycle. Edges use `from_node`/`to_node` instead of `from`/`to`, and
  nodes with no incoming edges are always the entry points. Nodes inherit the parent's model and tools by default;
  in place of per-node `model_provider` and `model_settings`, use `make_graph(tools=Choice([...], multiple=True),
  model=Choice([...]))`, which also sets node, edge, and timeout limits.
- **`workflow` → `graph`** — tasks become nodes and `dependencies` become edges, run in one call. Nothing is saved
  under `~/.strands/workflows`. `status`, partial workflow resumption (by re-running `start`), and `priority`
  have no equivalent. Ready nodes run in parallel. Per-task models and tools move to `make_graph` as above.
- **`agent_graph` → `graph`** — the execution model changes: `agent_graph` kept a network of agents running and passed
  messages between them, while the SDK tool runs each node once and propagates its output downstream. There are no
  `message`, `status`, or `stop` actions. `star` and `hierarchical` topologies map onto edges; cycles are not allowed,
  so `mesh` topologies fit the SDK `swarm` tool better.
- **`stop` → `stop`** — the run now ends with `stop_reason == "cancelled"` rather than the model's own `"tool_use"`,
  and `reason` becomes an optional `message` that is also the final assistant message, so `str(result)` returns it
  where it used to be empty. If you branch on `stop_reason` or read `request_state["stop_event_loop"]`, update those
  checks. The message is capped at 4096 characters by default.
- **`use_aws` → AWS MCP Server** — AWS maintains it as a managed remote server, reached through the
  `mcp-proxy-for-aws` proxy, and its `aws___call_aws` tool covers the same API surface as this tool. Write access is
  controlled by IAM, or the proxy's `--read-only` flag, rather than this tool's confirmation prompt. The self-hosted
  [AWS API MCP server](https://github.com/awslabs/mcp/tree/main/src/aws-api-mcp-server) still works but is superseded;
  see its [migration guide](https://github.com/awslabs/mcp/blob/main/src/aws-api-mcp-server/MIGRATION.md) and the
  [AWS MCP Server docs](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/mcp-server.html).
- **`mongodb_memory` → custom `MemoryStore`** — no store ships for MongoDB, but one only needs `search` (a
  `$vectorSearch` aggregation) and `add` (insert a document) over [`pymongo`](https://pymongo.readthedocs.io/).
  `retrieve` then becomes `search_memory` and `record` becomes `add_memory` (with `writable=True` on the store and
  `add_tool_config=True` on `MemoryManager`); bind the namespace in the store, not the tool call, to keep tenants
  isolated. For `list`, `get`, and `delete`, call `pymongo` directly.
