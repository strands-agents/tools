# Apify Core Tools

The Apify core tools (`apify.py`) provide the foundational building blocks for interacting with the [Apify](https://apify.com) platform from Strands Agents. These generic tools let you run any [Actor](https://apify.com/store) by ID, fetch Dataset results, and scrape individual URLs.

For higher-level, domain-specific tools see:
- [Apify Social Media Tools](apify_social_media_tool.md) — simplified wrappers for Instagram, LinkedIn, Twitter/X, TikTok, and Facebook scrapers
- [Apify Search Tools](apify_search_tool.md) — simplified wrappers for Google Search, Google Maps, YouTube, web crawling, and e-commerce scrapers

## Installation

```bash
pip install strands-agents-tools[apify]
```

## Configuration

Set your Apify API token as an environment variable:

```bash
export APIFY_API_TOKEN=apify_api_your_token_here
```

Get your token from the [Apify Console](https://console.apify.com/account/integrations) → Settings → API & Integrations → Personal API tokens.

## Usage

```python
from strands import Agent
from strands_tools import apify

agent = Agent(tools=[
    apify.apify_run_actor,
    apify.apify_scrape_url,
    apify.apify_get_dataset_items,
    apify.apify_run_actor_and_get_dataset,
])
```

### Scrape a URL

The simplest way to extract content from any web page. Uses the [Website Content Crawler](https://apify.com/apify/website-content-crawler) Actor under the hood and returns the page content as Markdown:

```python
content = agent.tool.apify_scrape_url(url="https://example.com")
```

### Run an Actor

Execute any Actor from the [Apify Store](https://apify.com/store) by its ID. The call blocks until the Actor Run finishes or the timeout is reached:

```python
result = agent.tool.apify_run_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
    timeout_secs=300,
)
```

The result is a JSON string containing run metadata: `run_id`, `status`, `dataset_id`, `started_at`, and `finished_at`.

### Run an Actor and Get Results

Combine running an Actor and fetching its Dataset results in a single call:

```python
result = agent.tool.apify_run_actor_and_get_dataset(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
    dataset_items_limit=50,
)
```

### Fetch Dataset Items

Retrieve results from a Dataset by its ID. Useful after running an Actor to get the structured results separately, or to access any existing Dataset:

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

**Returns:** Markdown content of the scraped page as a plain string.

### apify_run_actor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `actor_id` | string | Yes | — | Actor identifier (e.g., `apify/website-content-crawler`) |
| `run_input` | dict | No | `{}` | JSON-serializable input for the Actor |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor Run to finish |
| `memory_mbytes` | int | No | Actor default | Memory allocation in MB for the Actor Run |

**Returns:** JSON string with run metadata: `run_id`, `status`, `dataset_id`, `started_at`, `finished_at`.

### apify_get_dataset_items

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_id` | string | Yes | — | The Apify Dataset ID to fetch items from |
| `limit` | int | No | 100 | Maximum number of items to return |
| `offset` | int | No | 0 | Number of items to skip for pagination |

**Returns:** JSON string containing an array of Dataset items.

### apify_run_actor_and_get_dataset

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `actor_id` | string | Yes | — | Actor identifier (e.g., `apify/website-content-crawler`) |
| `run_input` | dict | No | `{}` | JSON-serializable input for the Actor |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor Run to finish |
| `memory_mbytes` | int | No | Actor default | Memory allocation in MB for the Actor Run |
| `dataset_items_limit` | int | No | 100 | Maximum number of Dataset items to return |

**Returns:** JSON string with run metadata plus an `items` array containing the Dataset results.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `APIFY_API_TOKEN environment variable is not set` | Token not configured | Set the `APIFY_API_TOKEN` environment variable |
| `apify-client package is required` | Optional dependency not installed | Run `pip install strands-agents-tools[apify]` |
| `Actor ... finished with status FAILED` | Actor execution error | Check Actor input parameters and run logs in the [Apify Console](https://console.apify.com) |
| `Actor ... finished with status TIMED-OUT` | Timeout too short for the workload | Increase the `timeout_secs` parameter |
| `No content returned for URL` | Website Content Crawler returned empty results | Verify the URL is accessible and returns content |

## References

- [Strands Agents Tools](https://strandsagents.com/latest/user-guide/concepts/tools/tools_overview/)
- [Apify Platform](https://apify.com)
- [Apify API Documentation](https://docs.apify.com/api/v2)
- [Apify Store](https://apify.com/store)
- [Apify Python Client](https://docs.apify.com/api/client/python/docs)
