# Apify

The Apify tools (`apify.py`) enable [Strands Agents](https://strandsagents.com/) to interact with the [Apify](https://apify.com) platform — running any [Actor](https://apify.com/store) or [task](https://docs.apify.com/platform/actors/running/tasks) by ID, fetching dataset results, and scraping individual URLs.

## Installation

```bash
pip install strands-agents-tools[apify]
```

## Configuration

Set your Apify API token as an environment variable:

```bash
export APIFY_API_TOKEN=apify_api_your_token_here
```

Get your token from [Apify Console](https://console.apify.com/account/integrations) → Settings → API & Integrations → Personal API tokens.

## Usage

Register all core tools at once:

```python
from strands import Agent
from strands_tools.apify import APIFY_CORE_TOOLS

agent = Agent(tools=APIFY_CORE_TOOLS)
```

Or pick individual tools:

```python
from strands import Agent
from strands_tools import apify

agent = Agent(tools=[
    apify.apify_run_actor,
    apify.apify_scrape_url,
])
```

### Scrape a URL

The simplest way to extract content from any web page. Uses the [Website Content Crawler](https://apify.com/apify/website-content-crawler) Actor under the hood and returns the page content as Markdown:

```python
content = agent.tool.apify_scrape_url(url="https://example.com")
```

### Run an Actor

Execute any Actor from [Apify Store](https://apify.com/store) by its ID. The call blocks until the Actor run finishes or the timeout is reached:

```python
result = agent.tool.apify_run_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
    timeout_secs=300,
)
```

The result is a JSON string containing run metadata: `run_id`, `status`, `dataset_id`, `started_at`, and `finished_at`.

### Run an Actor and Get Results

Combine running an Actor and fetching its dataset results in a single call:

```python
result = agent.tool.apify_run_actor_and_get_dataset(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
    dataset_items_limit=50,
)
```

### Run a task

Execute a saved [Actor task](https://docs.apify.com/platform/actors/running/tasks) — a pre-configured Actor with preset inputs. Use this when a task has already been set up in Apify Console:

```python
result = agent.tool.apify_run_task(
    task_id="user~my-task",
    task_input={"query": "override input"},
    timeout_secs=300,
)
```

The result is a JSON string containing run metadata: `run_id`, `status`, `dataset_id`, `started_at`, and `finished_at`.

### Run a task and get results

Combine running a task and fetching its dataset results in a single call:

```python
result = agent.tool.apify_run_task_and_get_dataset(
    task_id="user~my-task",
    dataset_items_limit=50,
)
```

### Fetch dataset items

Retrieve results from a dataset by its ID. Useful after running an Actor to get the structured results separately, or to access any existing dataset:

```python
items = agent.tool.apify_get_dataset_items(
    dataset_id="abc123",
    limit=100,
    offset=0,
)
```

## Tool Parameters

### apify_scrape_url

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | — | The URL to scrape |
| `timeout_secs` | int | No | 120 | Maximum time in seconds to wait for scraping to finish |
| `crawler_type` | string | No | `"cheerio"` | Crawler engine to use. One of `"cheerio"` (fastest, no JS rendering), `"playwright:adaptive"` (fast, renders JS if present), or `"playwright:firefox"` (reliable, renders JS, best at avoiding blocking but slower) |

**Returns:** Markdown content of the scraped page as a plain string.

### apify_run_actor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `actor_id` | string | Yes | — | Actor identifier (e.g., `apify/website-content-crawler`) |
| `run_input` | dict | No | None | JSON-serializable input for the Actor |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run to finish |
| `memory_mbytes` | int | No | None | Memory allocation in MB for the Actor run (uses Actor default if not set) |
| `build` | string | No | None | Actor build tag or number to run a specific version (uses latest build if not set) |

**Returns:** JSON string with run metadata: `run_id`, `status`, `dataset_id`, `started_at`, `finished_at`.

### apify_run_task

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task_id` | string | Yes | — | Task identifier (e.g., `user~my-task` or a task ID) |
| `task_input` | dict | No | None | JSON-serializable input to override the task's default input |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the task run to finish |
| `memory_mbytes` | int | No | None | Memory allocation in MB for the task run (uses task default if not set) |

**Returns:** JSON string with run metadata: `run_id`, `status`, `dataset_id`, `started_at`, `finished_at`.

### apify_run_task_and_get_dataset

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task_id` | string | Yes | — | Task identifier (e.g., `user~my-task` or a task ID) |
| `task_input` | dict | No | None | JSON-serializable input to override the task's default input |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the task run to finish |
| `memory_mbytes` | int | No | None | Memory allocation in MB for the task run (uses task default if not set) |
| `dataset_items_limit` | int | No | 100 | Maximum number of dataset items to return |
| `dataset_items_offset` | int | No | 0 | Number of dataset items to skip for pagination |

**Returns:** JSON string with run metadata plus an `items` array containing the dataset results.

### apify_get_dataset_items

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_id` | string | Yes | — | The Apify dataset ID to fetch items from |
| `limit` | int | No | 100 | Maximum number of items to return |
| `offset` | int | No | 0 | Number of items to skip for pagination |

**Returns:** JSON string containing an array of dataset items.

### apify_run_actor_and_get_dataset

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `actor_id` | string | Yes | — | Actor identifier (e.g., `apify/website-content-crawler`) |
| `run_input` | dict | No | None | JSON-serializable input for the Actor |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run to finish |
| `memory_mbytes` | int | No | None | Memory allocation in MB for the Actor run (uses Actor default if not set) |
| `build` | string | No | None | Actor build tag or number to run a specific version (uses latest build if not set) |
| `dataset_items_limit` | int | No | 100 | Maximum number of dataset items to return |
| `dataset_items_offset` | int | No | 0 | Number of dataset items to skip for pagination |

**Returns:** JSON string with run metadata plus an `items` array containing the dataset results.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `APIFY_API_TOKEN environment variable is not set` | Token not configured | Set the `APIFY_API_TOKEN` environment variable |
| `apify-client package is required` | Optional dependency not installed | Run `pip install strands-agents-tools[apify]` |
| `Actor ... finished with status FAILED` | Actor execution error | Check Actor input parameters and run logs in [Apify Console](https://console.apify.com) |
| `Task ... finished with status FAILED` | Task execution error | Check task configuration and run logs in [Apify Console](https://console.apify.com) |
| `Actor/task ... finished with status TIMED-OUT` | Timeout too short for the workload | Increase the `timeout_secs` parameter |
| `Task ... returned no run data` | Task `call()` returned `None` (wait timeout) | Increase the `timeout_secs` parameter |
| `No content returned for URL` | Website Content Crawler returned empty results | Verify the URL is accessible and returns content |

## References

- [Strands Agents Tools](https://strandsagents.com/latest/user-guide/concepts/tools/tools_overview/)
- [Apify Platform](https://apify.com)
- [Apify API Documentation](https://docs.apify.com/api/v2)
- [Apify Store](https://apify.com/store)
- [Apify Python Client](https://docs.apify.com/api/client/python/docs)
