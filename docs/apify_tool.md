# Apify

The Apify tools (`apify.py`) enable [Strands Agents](https://strandsagents.com/) to interact with the [Apify](https://apify.com) platform — running any [Actor](https://apify.com/store) or [task](https://docs.apify.com/platform/actors/running/tasks) by ID, fetching dataset results, scraping individual URLs, and scraping popular social media platforms.

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

Register all social media tools at once:

```python
from strands import Agent
from strands_tools.apify import APIFY_SOCIAL_TOOLS

agent = Agent(tools=APIFY_SOCIAL_TOOLS)
```

Or combine both, or pick individual tools:

```python
from strands import Agent
from strands_tools.apify import APIFY_CORE_TOOLS
from strands_tools import apify

agent = Agent(tools=[
    *APIFY_CORE_TOOLS,
    apify.apify_instagram_scraper,
    apify.apify_twitter_scraper,
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

**Returns:** Markdown content of the scraped page as a plain string.

### apify_run_actor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `actor_id` | string | Yes | — | Actor identifier (e.g., `apify/website-content-crawler`) |
| `run_input` | dict | No | None | JSON-serializable input for the Actor |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run to finish |
| `memory_mbytes` | int | No | None | Memory allocation in MB for the Actor run (uses Actor default if not set) |

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
| `dataset_items_limit` | int | No | 100 | Maximum number of dataset items to return |

**Returns:** JSON string with run metadata plus an `items` array containing the dataset results.

## Social Media Scraping

The Apify module includes simplified wrappers for 7 popular social media scraping Actors. Each tool exposes a small set of LLM-friendly parameters instead of the full Actor input schema, runs the Actor synchronously, and returns the dataset results as JSON.

All social media tools default to returning at most 20 results (`results_limit=20`) to keep the response concise for LLM consumption.

### Scrape Instagram

Search for Instagram profiles, hashtags, or places, or scrape specific Instagram URLs. Uses the [Instagram Scraper](https://apify.com/apify/instagram-scraper) Actor.

```python
# Search for a hashtag and get posts
result = agent.tool.apify_instagram_scraper(
    search_query="travel", search_type="hashtag", results_type="posts", results_limit=10,
)

# Get profile details (metadata only)
result = agent.tool.apify_instagram_scraper(
    urls=["https://www.instagram.com/apify/"], results_type="details",
)

# A URL passed as search_query is auto-detected
result = agent.tool.apify_instagram_scraper(search_query="https://www.instagram.com/apify/")
```

### Scrape LinkedIn profile posts

Fetch recent posts from a LinkedIn profile. Accepts a profile URL or bare username. Uses the [LinkedIn Profile Posts](https://apify.com/apimaestro/linkedin-profile-posts) Actor.

```python
result = agent.tool.apify_linkedin_profile_posts(
    profile_url="https://www.linkedin.com/in/neal-mohan",
    results_limit=15,
)
```

### Search LinkedIn profiles

Find people on LinkedIn by keywords with optional filters for location and job title. Uses the [LinkedIn Profile Search](https://apify.com/harvestapi/linkedin-profile-search) Actor.

```python
# Simple keyword search
result = agent.tool.apify_linkedin_profile_search(
    search_query="software engineer",
    results_limit=25,
)

# Search with filters and full profile details
result = agent.tool.apify_linkedin_profile_search(
    search_query="marketing manager",
    locations=["San Francisco", "New York"],
    current_job_titles=["Marketing Manager", "Head of Marketing"],
    profile_scraper_mode="Full",
)
```

### Get LinkedIn profile details

Retrieve full profile information including work experience, education, and skills. Accepts a profile URL or bare username. No LinkedIn account or cookies required. Uses the [LinkedIn Profile Detail](https://apify.com/apimaestro/linkedin-profile-detail) Actor.

```python
result = agent.tool.apify_linkedin_profile_detail(
    profile_url="https://www.linkedin.com/in/neal-mohan",
    include_email=True,
)
```

### Scrape Twitter/X

Search for tweets, scrape by handle, or scrape specific URLs. Supports [Twitter advanced search](https://github.com/igorbrigadir/twitter-advanced-search) syntax. Uses the [Twitter Scraper Lite](https://apify.com/apidojo/twitter-scraper-lite) Actor.

```python
# Search tweets with sort order
result = agent.tool.apify_twitter_scraper(
    search_query="from:NASA", results_limit=30, sort="Latest",
)

# Scrape by Twitter handles
result = agent.tool.apify_twitter_scraper(
    twitter_handles=["NASA", "SpaceX"], tweet_language="en",
)

# Scrape specific tweet URLs
result = agent.tool.apify_twitter_scraper(
    urls=["https://x.com/elonmusk/status/1728108619189874825"],
)
```

### Scrape TikTok

Search by keyword, hashtag, profile, or specific post URL. Uses the [TikTok Scraper](https://apify.com/clockworks/tiktok-scraper) Actor.

```python
# Search by keyword
result = agent.tool.apify_tiktok_scraper(search_query="cooking", results_limit=15)

# Scrape videos by hashtag
result = agent.tool.apify_tiktok_scraper(hashtags=["fyp", "cooking"])

# Scrape videos from specific profiles
result = agent.tool.apify_tiktok_scraper(profiles=["charlidamelio"])

# Scrape specific post URLs
result = agent.tool.apify_tiktok_scraper(
    urls=["https://www.tiktok.com/@user/video/123"],
)
```

### Scrape Facebook posts

Scrape posts from a Facebook page or profile. Optionally filter by date. Uses the [Facebook Posts Scraper](https://apify.com/apify/facebook-posts-scraper) Actor.

```python
result = agent.tool.apify_facebook_posts_scraper(
    page_url="https://www.facebook.com/apify",
    results_limit=10,
    only_posts_newer_than="2024-01-01",
)
```

## Social Media Tool Parameters

### apify_instagram_scraper

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_query` | string | No* | None | Username, hashtag, or keyword to search. URLs are auto-detected and routed to direct scraping. |
| `urls` | list[str] | No* | None | One or more Instagram URLs to scrape directly |
| `results_type` | string | No | `"posts"` | What to scrape: `"posts"`, `"comments"`, or `"details"` (profile metadata) |
| `results_limit` | int | No | 20 | Maximum number of items per URL or search hit |
| `search_type` | string | No | `"hashtag"` | What to search for: `"hashtag"`, `"user"`, or `"place"` |
| `search_limit` | int | No | 10 | How many search results (hashtags/users/places) to process |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

\* At least one of `search_query` or `urls` is required.

**Returns:** JSON string with run metadata and an `items` array containing scraped Instagram data.

### apify_linkedin_profile_posts

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `profile_url` | string | Yes | — | LinkedIn profile URL or bare username |
| `results_limit` | int | No | 20 | Maximum number of posts to return (capped at 100) |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

**Returns:** JSON string with run metadata and an `items` array containing LinkedIn post data.

### apify_linkedin_profile_search

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_query` | string | Yes | — | Search keywords (e.g., `"software engineer"`, `"marketing manager"`) |
| `results_limit` | int | No | 20 | Maximum number of profiles to return |
| `locations` | list[str] | No | None | Filter by locations (e.g., `["San Francisco", "New York"]`) |
| `current_job_titles` | list[str] | No | None | Filter by current job titles (e.g., `["Software Engineer"]`) |
| `profile_scraper_mode` | string | No | `"Short"` | `"Short"` for basic data, `"Full"` for complete details (experience, education, skills) |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

**Returns:** JSON string with run metadata and an `items` array containing matched LinkedIn profiles.

### apify_linkedin_profile_detail

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `profile_url` | string | Yes | — | LinkedIn profile URL or bare username |
| `include_email` | bool | No | False | Include email address if publicly available |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

**Returns:** JSON string with run metadata and an `items` array containing detailed profile data (work experience, education, certifications, location, and optionally email).

### apify_twitter_scraper

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_query` | string | No* | None | Search query. Supports [Twitter advanced search](https://github.com/igorbrigadir/twitter-advanced-search) operators (`from:`, `#hashtag`, `min_faves:`, `since:`, `until:`) |
| `urls` | list[str] | No* | None | Specific tweet, profile, search, or list URLs to scrape |
| `twitter_handles` | list[str] | No* | None | Twitter handles to scrape (without `@`, e.g. `["NASA", "elonmusk"]`) |
| `results_limit` | int | No | 20 | Maximum number of tweets to return |
| `sort` | string | No | `"Latest"` | Sort order: `"Latest"` (chronological) or `"Top"` (popular) |
| `tweet_language` | string | No | None | ISO 639-1 language code (e.g. `"en"`, `"es"`) |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

\* At least one of `search_query`, `urls`, or `twitter_handles` is required.

**Returns:** JSON string with run metadata and an `items` array containing scraped tweet data.

### apify_tiktok_scraper

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_query` | string | No* | None | Keyword to search across videos and profiles |
| `hashtags` | list[str] | No* | None | Hashtags to scrape videos from (without `#`, e.g. `["fyp", "cooking"]`) |
| `profiles` | list[str] | No* | None | TikTok usernames to scrape videos from |
| `urls` | list[str] | No* | None | Specific TikTok post URLs to scrape |
| `results_limit` | int | No | 20 | Maximum number of videos per hashtag, profile, or search |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

\* At least one of `search_query`, `hashtags`, `profiles`, or `urls` is required.

**Returns:** JSON string with run metadata and an `items` array containing scraped TikTok video data.

### apify_facebook_posts_scraper

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page_url` | string | Yes | — | Facebook page or profile URL to scrape posts from |
| `results_limit` | int | No | 20 | Maximum number of posts to return |
| `only_posts_newer_than` | string | No | None | Only return posts newer than this date (e.g. `"2024-01-01"`, `"1 week ago"`) |
| `timeout_secs` | int | No | 300 | Maximum time in seconds to wait for the Actor run |

**Returns:** JSON string with run metadata and an `items` array containing scraped Facebook post data.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `APIFY_API_TOKEN environment variable is not set` | Token not configured | Set the `APIFY_API_TOKEN` environment variable |
| `apify-client package is required` | Optional dependency not installed | Run `pip install strands-agents-tools[apify]` |
| `Actor ... finished with status FAILED` | Actor execution error | Check Actor input parameters and run logs in [Apify Console](https://console.apify.com) |
| `Task ... finished with status FAILED` | task execution error | Check task configuration and run logs in [Apify Console](https://console.apify.com) |
| `Actor/task ... finished with status TIMED-OUT` | Timeout too short for the workload | Increase the `timeout_secs` parameter |
| `Task ... returned no run data` | task `call()` returned `None` (wait timeout) | Increase the `timeout_secs` parameter |
| `No content returned for URL` | Website Content Crawler returned empty results | Verify the URL is accessible and returns content |
| `Provide at least one of 'search_query' or 'urls'` | Neither parameter was provided to a social media tool that requires one | Pass `search_query`, `urls`, or both |

## References

- [Strands Agents Tools](https://strandsagents.com/latest/user-guide/concepts/tools/tools_overview/)
- [Apify Platform](https://apify.com)
- [Apify API Documentation](https://docs.apify.com/api/v2)
- [Apify Store](https://apify.com/store)
- [Apify Python Client](https://docs.apify.com/api/client/python/docs)

### Social media Actors used

- [Instagram Scraper](https://apify.com/apify/instagram-scraper)
- [LinkedIn Profile Posts](https://apify.com/apimaestro/linkedin-profile-posts)
- [LinkedIn Profile Search](https://apify.com/harvestapi/linkedin-profile-search)
- [LinkedIn Profile Detail](https://apify.com/apimaestro/linkedin-profile-detail)
- [Twitter Scraper Lite](https://apify.com/apidojo/twitter-scraper-lite)
- [TikTok Scraper](https://apify.com/clockworks/tiktok-scraper)
- [Facebook Posts Scraper](https://apify.com/apify/facebook-posts-scraper)
