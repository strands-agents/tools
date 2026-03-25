"""Social media scraping tools for Strands Agents via the Apify platform.

This module provides simplified wrappers around popular social media scraping
Actors on Apify. Each tool exposes a small set of LLM-friendly parameters
instead of the full Actor input schema, runs the Actor synchronously, and
returns the dataset results as JSON.

Available Tools:
---------------
- apify_instagram_scraper: Scrape Instagram profiles, posts, or hashtags
- apify_linkedin_profile_posts: Scrape posts from a LinkedIn profile
- apify_linkedin_profile_search: Search for LinkedIn profiles by keywords
- apify_linkedin_profile_detail: Get detailed LinkedIn profile information
- apify_twitter_scraper: Scrape tweets from Twitter/X
- apify_tiktok_scraper: Scrape TikTok videos, profiles, or hashtags
- apify_facebook_posts_scraper: Scrape posts from Facebook pages

Setup Requirements:
------------------
1. Create an Apify account at https://apify.com
2. Obtain your API token: Apify Console > Settings > API & Integrations > Personal API tokens
3. Install the optional dependency: pip install strands-agents-tools[apify]
4. Set the environment variable:
   APIFY_API_TOKEN=your_api_token_here

Usage Examples:
--------------
Register all social media tools at once via the preset list:

```python
from strands import Agent
from strands_tools.apify_social import APIFY_SOCIAL_TOOLS

agent = Agent(tools=APIFY_SOCIAL_TOOLS)
```

Or pick individual tools for a smaller LLM tool surface:

```python
from strands import Agent
from strands_tools import apify_social

agent = Agent(tools=[
    apify_social.apify_instagram_scraper,
    apify_social.apify_twitter_scraper,
])
```
"""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from strands import tool

from strands_tools.apify import (
    DEFAULT_DATASET_ITEMS_LIMIT,
    DEFAULT_TIMEOUT_SECS,
    ApifyToolClient,
    _check_dependency,
    _error_result,
    _success_result,
)

DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT = 20
INSTAGRAM_SCRAPER = "apify/instagram-scraper"
LINKEDIN_PROFILE_POSTS = "apimaestro/linkedin-profile-posts"
LINKEDIN_PROFILE_SEARCH = "harvestapi/linkedin-profile-search"
LINKEDIN_PROFILE_DETAIL = "apimaestro/linkedin-profile-detail"
TWITTER_SCRAPER = "apidojo/twitter-scraper-lite"
TIKTOK_SCRAPER = "clockworks/tiktok-scraper"
FACEBOOK_POSTS_SCRAPER = "apify/facebook-posts-scraper"
_MISSING_SEARCH_OR_URLS = "Provide at least one of 'search_query' or 'urls'."


def _extract_linkedin_username(profile_url: str) -> str:
    """Extract a LinkedIn username from a profile URL, or return the value as-is if already a username."""
    parsed = urlparse(profile_url)
    if parsed.netloc and "linkedin.com" in parsed.netloc:
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 2 and parts[0] == "in":
            return parts[1]
    return profile_url


def _social_media_result(
    actor_name: str,
    client: ApifyToolClient,
    run_input: Dict[str, Any],
    actor_id: str,
    timeout_secs: int,
    results_limit: int,
) -> Dict[str, Any]:
    """Shared execution logic for social media scraper tools."""
    result = client.run_actor_and_get_dataset(
        actor_id=actor_id,
        run_input=run_input,
        timeout_secs=timeout_secs,
        dataset_items_limit=results_limit,
    )
    return _success_result(
        text=json.dumps(result, indent=2, default=str),
        panel_body=(
            f"[green]{actor_name} completed[/green]\n"
            f"Actor: {actor_id}\n"
            f"Run ID: {result['run_id']}\n"
            f"Status: {result['status']}\n"
            f"Items returned: {len(result['items'])}"
        ),
        panel_title=f"Apify: {actor_name}",
    )


@tool
def apify_instagram_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    search_type: str = "user",
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape Instagram profiles, posts, reels, or hashtags.

    Provide either a search query to discover content or direct Instagram URLs to scrape.
    Supports searching by user profile, hashtag, or place.

    Args:
        search_query: Username, hashtag, or keyword to search for on Instagram.
            If the value looks like an Instagram URL it is treated as a direct URL instead.
        urls: One or more Instagram URLs to scrape directly (profiles, posts, reels, etc.).
        results_limit: Maximum number of results to return. Defaults to 20.
        search_type: What to search for: "user", "hashtag", or "place". Defaults to "user".
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped Instagram data items.
    """
    try:
        _check_dependency()
        if not search_query and not urls:
            raise ValueError(_MISSING_SEARCH_OR_URLS)

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {"resultsLimit": results_limit}

        if urls:
            run_input["directUrls"] = urls
        elif search_query and ("instagram.com" in search_query or search_query.startswith("http")):
            run_input["directUrls"] = [search_query]
        else:
            run_input["search"] = search_query
            run_input["searchType"] = search_type

        return _social_media_result(
            actor_name="Instagram Scraper",
            client=client,
            run_input=run_input,
            actor_id=INSTAGRAM_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_instagram_scraper")


@tool
def apify_linkedin_profile_posts(
    profile_url: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape posts from a LinkedIn profile.

    Accepts a LinkedIn profile URL (e.g. "https://www.linkedin.com/in/username") or
    a bare username. Returns the most recent posts from that profile.

    Args:
        profile_url: LinkedIn profile URL or username to scrape posts from.
        results_limit: Maximum number of posts to return (1-100). Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped LinkedIn post data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        username = _extract_linkedin_username(profile_url)
        run_input: Dict[str, Any] = {
            "username": username,
            "limit": min(results_limit, 100),
        }
        return _social_media_result(
            actor_name="LinkedIn Profile Posts",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_POSTS,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_posts")


@tool
def apify_linkedin_profile_search(
    search_query: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Search for LinkedIn profiles by keywords.

    Find people on LinkedIn using keywords like job titles, skills, company names,
    or locations (e.g. "software engineer San Francisco").

    Args:
        search_query: Search keywords to find LinkedIn profiles.
        results_limit: Maximum number of profiles to return. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing matched LinkedIn profile data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "searchQuery": search_query,
            "maxItems": results_limit,
        }
        return _social_media_result(
            actor_name="LinkedIn Profile Search",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_SEARCH,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_search")


@tool
def apify_linkedin_profile_detail(
    profile_url: str,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Get detailed information from a LinkedIn profile.

    Accepts a LinkedIn profile URL (e.g. "https://www.linkedin.com/in/username") or
    a bare username. Returns full profile details including work experience, education,
    skills, and more.

    Args:
        profile_url: LinkedIn profile URL or username to scrape.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing detailed LinkedIn profile data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        username = _extract_linkedin_username(profile_url)
        run_input: Dict[str, Any] = {"username": username}
        return _social_media_result(
            actor_name="LinkedIn Profile Detail",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_DETAIL,
            timeout_secs=timeout_secs,
            results_limit=DEFAULT_DATASET_ITEMS_LIMIT,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_detail")


@tool
def apify_twitter_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape tweets from Twitter/X by search query or specific URLs.

    Supports Twitter advanced search syntax (e.g. "from:NASA", "#AI min_faves:100").
    Provide either a search query or direct tweet/profile/list URLs.

    Args:
        search_query: Search query or hashtag to find tweets. Supports Twitter advanced search.
        urls: Specific tweet, profile, or list URLs to scrape.
        results_limit: Maximum number of tweets to return. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped tweet data.
    """
    try:
        _check_dependency()
        if not search_query and not urls:
            raise ValueError(_MISSING_SEARCH_OR_URLS)

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {"maxItems": results_limit}

        if search_query:
            run_input["searchTerms"] = [search_query]
        if urls:
            run_input["startUrls"] = urls

        return _social_media_result(
            actor_name="Twitter Scraper",
            client=client,
            run_input=run_input,
            actor_id=TWITTER_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_twitter_scraper")


@tool
def apify_tiktok_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape TikTok videos, profiles, or hashtags.

    Provide a search query to find TikTok content or direct post URLs to scrape.
    The search applies to both videos and profiles.

    Args:
        search_query: Search query, username, or hashtag to find TikTok content.
        urls: Specific TikTok post URLs to scrape.
        results_limit: Maximum number of results per query. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped TikTok data.
    """
    try:
        _check_dependency()
        if not search_query and not urls:
            raise ValueError(_MISSING_SEARCH_OR_URLS)

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {"resultsPerPage": results_limit}

        if search_query:
            run_input["searchQueries"] = [search_query]
        if urls:
            run_input["postURLs"] = urls

        return _social_media_result(
            actor_name="TikTok Scraper",
            client=client,
            run_input=run_input,
            actor_id=TIKTOK_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_tiktok_scraper")


@tool
def apify_facebook_posts_scraper(
    page_url: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape posts from a Facebook page or profile.

    Provide a Facebook page or profile URL to scrape its posts.

    Args:
        page_url: Facebook page or profile URL to scrape posts from.
        results_limit: Maximum number of posts to return. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped Facebook post data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "startUrls": [{"url": page_url}],
            "resultsLimit": results_limit,
        }
        return _social_media_result(
            actor_name="Facebook Posts Scraper",
            client=client,
            run_input=run_input,
            actor_id=FACEBOOK_POSTS_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_facebook_posts_scraper")


# Pre-built list of all social media tools for convenient agent registration.
# Usage: Agent(tools=APIFY_SOCIAL_TOOLS)
APIFY_SOCIAL_TOOLS = [
    apify_instagram_scraper,
    apify_linkedin_profile_posts,
    apify_linkedin_profile_search,
    apify_linkedin_profile_detail,
    apify_twitter_scraper,
    apify_tiktok_scraper,
    apify_facebook_posts_scraper,
]
