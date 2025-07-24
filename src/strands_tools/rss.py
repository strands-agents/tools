import json
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import feedparser
import html2text
import requests
from strands import tool

# Configure logging and defaults
logger = logging.getLogger(__name__)
# Always use temporary directory for storage
DEFAULT_STORAGE_PATH = os.path.join(tempfile.gettempdir(), "strands_rss_feeds")
DEFAULT_MAX_ENTRIES = int(os.environ.get("STRANDS_RSS_MAX_ENTRIES", "100"))
DEFAULT_UPDATE_INTERVAL = int(os.environ.get("STRANDS_RSS_UPDATE_INTERVAL", "60"))  # minutes

# Create HTML to text converter
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = True
html_converter.body_width = 0


class RSSManager:
    """Manage RSS feed subscriptions, updates, and content retrieval."""

    def __init__(self):
        self.storage_path = os.environ.get("STRANDS_RSS_STORAGE_PATH", DEFAULT_STORAGE_PATH)
        os.makedirs(self.storage_path, exist_ok=True)

    def get_feed_file_path(self, feed_id: str) -> str:
        return os.path.join(self.storage_path, f"{feed_id}.json")

    def get_subscription_file_path(self) -> str:
        return os.path.join(self.storage_path, "subscriptions.json")

    def clean_html(self, html_content: str) -> str:
        return "" if not html_content else html_converter.handle(html_content)

    def format_entry(self, entry: Dict, include_content: bool = False) -> Dict:
        result = {
            "title": entry.get("title", "Untitled"),
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "Unknown date")),
            "author": entry.get("author", "Unknown author"),
        }

        # Add categories
        if "tags" in entry:
            result["categories"] = [tag.get("term", "") for tag in entry.tags if "term" in tag]
        elif "categories" in entry:
            result["categories"] = entry.get("categories", [])

        # Add content if requested
        if include_content:
            content = ""
            # Handle content as both attribute and dictionary key
            if "content" in entry:
                # Handle dictionary access
                if isinstance(entry["content"], list):
                    for item in entry["content"]:
                        if isinstance(item, dict) and "value" in item:
                            content = self.clean_html(item["value"])
                            break
                # Handle string content directly
                elif isinstance(entry["content"], str):
                    content = self.clean_html(entry["content"])
            # Handle summary and description fields
            if not content and "summary" in entry:
                content = self.clean_html(entry["summary"])
            if not content and "description" in entry:
                content = self.clean_html(entry["description"])
            result["content"] = content or "No content available"

        return result

    def generate_feed_id(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.rstrip("/").replace("/", "_") or "main"
        return f"{domain}{path}".replace(".", "_").lower()

    def load_subscriptions(self) -> Dict[str, Dict]:
        file_path = self.get_subscription_file_path()
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error parsing subscription file: {file_path}")
            return {}

    def save_subscriptions(self, subscriptions: Dict[str, Dict]) -> None:
        """Save subscriptions to JSON file with proper formatting."""
        file_path = self.get_subscription_file_path()
        with open(file_path, "w") as f:
            json.dump(subscriptions, f, indent=2)

    def load_feed_data(self, feed_id: str) -> Dict:
        file_path = self.get_feed_file_path(feed_id)
        if not os.path.exists(file_path):
            return {"entries": []}
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error parsing feed file: {file_path}")
            return {"entries": []}

    def save_feed_data(self, feed_id: str, data: Dict) -> None:
        with open(self.get_feed_file_path(feed_id), "w") as f:
            json.dump(data, f, indent=2)

    def fetch_feed(self, url: str, auth: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        # Initialize headers dictionary if not provided
        if headers is None:
            headers = {}
        # Handle case where headers might be a string (for backward compatibility with tests)
        elif isinstance(headers, str):
            headers = {"User-Agent": headers}

        # If using basic auth, make the request with headers and auth
        if auth and auth.get("type") == "basic":
            response = requests.get(url, headers=headers, auth=(auth.get("username", ""), auth.get("password", "")))
            return feedparser.parse(response.content)

        # For non-auth requests, extract User-Agent if present in headers
        user_agent = headers.get("User-Agent")
        return feedparser.parse(url, agent=user_agent)

    def update_feed(self, feed_id: str, subscriptions: Dict[str, Dict]) -> Dict:
        if feed_id not in subscriptions:
            return {"status": "error", "content": [{"text": f"Feed {feed_id} not found in subscriptions"}]}

        try:
            feed_info = subscriptions[feed_id]
            feed = self.fetch_feed(feed_info["url"], feed_info.get("auth"), feed_info.get("headers"))

            if not hasattr(feed, "entries"):
                return {"status": "error", "content": [{"text": f"Could not parse feed from {feed_info['url']}"}]}

            # Process feed data
            feed_data = self.load_feed_data(feed_id)
            existing_ids = {entry.get("id", entry.get("link")) for entry in feed_data.get("entries", [])}

            # Update metadata
            feed_data.update(
                {
                    "title": getattr(feed.feed, "title", feed_info["url"]),
                    "description": getattr(feed.feed, "description", ""),
                    "link": getattr(feed.feed, "link", feed_info["url"]),
                    "last_updated": datetime.now().isoformat(),
                }
            )

            # Add new entries
            new_entries = []
            for entry in feed.entries:
                entry_id = entry.get("id", entry.get("link"))
                if entry_id and entry_id not in existing_ids:
                    entry_data = self.format_entry(entry, include_content=True)
                    entry_data["id"] = entry_id
                    new_entries.append(entry_data)

            # Update entries and save
            feed_data["entries"] = (new_entries + feed_data.get("entries", []))[:DEFAULT_MAX_ENTRIES]
            self.save_feed_data(feed_id, feed_data)

            # Update subscription metadata
            subscriptions[feed_id]["title"] = feed_data["title"]
            subscriptions[feed_id]["last_updated"] = feed_data["last_updated"]
            self.save_subscriptions(subscriptions)

            return {
                "feed_id": feed_id,
                "title": feed_data["title"],
                "new_entries": len(new_entries),
                "total_entries": len(feed_data["entries"]),
            }

        except Exception as e:
            logger.error(f"Error updating feed {feed_id}: {str(e)}")
            return {"status": "error", "content": [{"text": f"Error updating feed {feed_id}: {str(e)}"}]}


# Initialize RSS manager
rss_manager = RSSManager()


@tool
def rss(
    action: str,
    url: Optional[str] = None,
    feed_id: Optional[str] = None,
    max_entries: int = 10,
    include_content: bool = False,
    query: Optional[str] = None,
    category: Optional[str] = None,
    update_interval: Optional[int] = None,
    auth_username: Optional[str] = None,
    auth_password: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Union[List[Dict], Dict]:
    """
    Interact with RSS feeds - fetch, subscribe, search, and manage feeds.

    Actions:
    - fetch: Get feed content from URL without subscribing
    - subscribe: Add a feed to your subscription list
    - unsubscribe: Remove a feed subscription
    - list: List all subscribed feeds
    - read: Read entries from a subscribed feed
    - update: Update feeds with new content
    - search: Find entries matching a query
    - categories: List all categories/tags

    Args:
        action: Action to perform (fetch, subscribe, unsubscribe, list, read, update, search, categories)
        url: URL of the RSS feed (for fetch and subscribe)
        feed_id: ID of a subscribed feed (for read/update/unsubscribe)
        max_entries: Maximum number of entries to return (default: 10)
        include_content: Whether to include full content (default: False)
        query: Search query for filtering entries
        category: Filter entries by category/tag
        update_interval: Update interval in minutes
        auth_username: Username for authenticated feeds
        auth_password: Password for authenticated feeds
        headers: Dictionary of HTTP headers to send with requests (e.g., {"User-Agent": "MyRSSReader/1.0"})
    """
    try:
        if action == "fetch":
            if not url:
                return {"status": "error", "content": [{"text": "URL is required for fetch action"}]}

            feed = rss_manager.fetch_feed(url, headers=headers)
            if not hasattr(feed, "entries"):
                return {"status": "error", "content": [{"text": f"Could not parse feed from {url}"}]}

            entries = [rss_manager.format_entry(entry, include_content) for entry in feed.entries[:max_entries]]
            return entries if entries else {"status": "error", "content": [{"text": "Feed contains no entries"}]}

        elif action == "subscribe":
            if not url:
                return {"status": "error", "content": [{"text": "URL is required for subscribe action"}]}

            feed_id = feed_id or rss_manager.generate_feed_id(url)
            subscriptions = rss_manager.load_subscriptions()

            if feed_id in subscriptions:
                return {"status": "error", "content": [{"text": f"Already subscribed to this feed with ID: {feed_id}"}]}

            # Create subscription
            subscription = {
                "url": url,
                "added_at": datetime.now().isoformat(),
                "update_interval": update_interval or DEFAULT_UPDATE_INTERVAL,
            }

            if auth_username and auth_password:
                subscription["auth"] = {"type": "basic", "username": auth_username, "password": auth_password}
            if headers:
                subscription["headers"] = headers

            subscriptions[feed_id] = subscription
            rss_manager.save_subscriptions(subscriptions)

            # Fetch initial data
            update_result = rss_manager.update_feed(feed_id, subscriptions)
            if "status" in update_result and update_result["status"] == "error":
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Subscribed with ID: {feed_id}, \
                            but error during fetch: {update_result['content'][0]['text']}"
                        }
                    ],
                }

            return {
                "status": "success",
                "content": [{"text": f"Subscribed to: {update_result.get('title', url)} with ID: {feed_id}"}],
            }

        elif action == "unsubscribe":
            if not feed_id:
                return {"status": "error", "content": [{"text": "feed_id is required for unsubscribe action"}]}

            subscriptions = rss_manager.load_subscriptions()
            if feed_id not in subscriptions:
                return {"status": "error", "content": [{"text": f"Not subscribed to feed with ID: {feed_id}"}]}

            feed_info = subscriptions.pop(feed_id)
            rss_manager.save_subscriptions(subscriptions)

            # Remove stored data file
            feed_file = rss_manager.get_feed_file_path(feed_id)
            if os.path.exists(feed_file):
                os.remove(feed_file)

            return {
                "status": "success",
                "content": [{"text": f"Unsubscribed from: {feed_info.get('title', feed_info.get('url', feed_id))}"}],
            }

        elif action == "list":
            subscriptions = rss_manager.load_subscriptions()
            if not subscriptions:
                return {"status": "error", "content": [{"text": "No subscribed feeds"}]}

            return [
                {
                    "feed_id": fid,
                    "title": info.get("title", info.get("url", "Unknown")),
                    "url": info.get("url", ""),
                    "last_updated": info.get("last_updated", "Never"),
                    "update_interval": info.get("update_interval", DEFAULT_UPDATE_INTERVAL),
                }
                for fid, info in subscriptions.items()
            ]

        elif action == "read":
            if not feed_id:
                return {"status": "error", "content": [{"text": "feed_id is required for read action"}]}

            subscriptions = rss_manager.load_subscriptions()
            if feed_id not in subscriptions:
                return {"status": "error", "content": [{"text": f"Not subscribed to feed with ID: {feed_id}"}]}

            feed_data = rss_manager.load_feed_data(feed_id)
            if not feed_data.get("entries"):
                return {"status": "error", "content": [{"text": f"No entries found for feed: {feed_id}"}]}

            entries = feed_data["entries"]
            if category:
                entries = [
                    entry
                    for entry in entries
                    if "categories" in entry and category.lower() in [c.lower() for c in entry["categories"]]
                ]

            return {
                "feed_id": feed_id,
                "title": feed_data.get("title", subscriptions[feed_id].get("url", "")),
                "entries": entries[:max_entries],
                "include_content": include_content,
            }

        elif action == "update":
            subscriptions = rss_manager.load_subscriptions()
            if not subscriptions:
                return {"status": "error", "content": [{"text": "No subscribed feeds to update"}]}

            if feed_id:
                if feed_id not in subscriptions:
                    return {"status": "error", "content": [{"text": f"Not subscribed to feed with ID: {feed_id}"}]}
                return rss_manager.update_feed(feed_id, subscriptions)
            else:
                return [rss_manager.update_feed(fid, subscriptions) for fid in subscriptions]

        elif action == "search":
            if not query:
                return {"status": "error", "content": [{"text": "query is required for search action"}]}

            subscriptions = rss_manager.load_subscriptions()
            if not subscriptions:
                return {"status": "error", "content": [{"text": "No subscribed feeds to search"}]}

            # Setup search pattern
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error:
                pattern = None

            # Track search results across all feeds
            results = []

            for fid in subscriptions:
                feed_data = rss_manager.load_feed_data(fid)
                feed_title = feed_data.get("title", subscriptions[fid].get("url", ""))

                for entry in feed_data.get("entries", []):
                    # Check for match in title or content
                    title_match = (
                        pattern.search(entry.get("title", ""))
                        if pattern
                        else query.lower() in entry.get("title", "").lower()
                    )

                    content_match = False
                    if include_content and not title_match:
                        content_match = (
                            pattern.search(entry.get("content", ""))
                            if pattern
                            else query.lower() in entry.get("content", "").lower()
                        )

                    if title_match or content_match:
                        results.append({"feed_id": fid, "feed_title": feed_title, "entry": entry})

                if len(results) >= max_entries:
                    # Break outer loop when we reach max_entries
                    break

            # Ensure we don't return more than max_entries
            results = results[:max_entries]

            return (
                results
                if results
                else {"status": "error", "content": [{"text": f"No entries found matching query: {query}"}]}
            )

        elif action == "categories":
            subscriptions = rss_manager.load_subscriptions()
            if not subscriptions:
                return {"status": "error", "content": [{"text": "No subscribed feeds"}]}

            all_categories: Set[str] = set()
            feed_categories: Dict[str, Set[str]] = {}

            for fid in subscriptions:
                feed_data = rss_manager.load_feed_data(fid)
                feed_title = feed_data.get("title", subscriptions[fid].get("url", ""))

                categories = set()
                for entry in feed_data.get("entries", []):
                    if "categories" in entry:
                        categories.update(entry["categories"])

                if categories:
                    all_categories.update(categories)
                    feed_categories[feed_title] = categories

            if not all_categories:
                return {"status": "error", "content": [{"text": "No categories found across feeds"}]}

            return {
                "all_categories": sorted(list(all_categories)),
                "feed_categories": {feed: sorted(list(cats)) for feed, cats in feed_categories.items()},
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action '{action}'. Valid actions: \
            fetch, subscribe, unsubscribe, list, read, update, search, categories"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"RSS tool error: {str(e)}")
        return {"status": "error", "content": [{"text": f"{str(e)}"}]}
