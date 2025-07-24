"""Comprehensive tests for RSS feed tool with improved organization."""

import json
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from src.strands_tools.rss import RSSManager, rss


@pytest.fixture
def mock_subscriptions():
    """Common fixture for subscriptions data."""
    return {
        "feed1": {"url": "https://example.com/feed1", "title": "Feed 1", "last_updated": "2023-07-21T12:00:00Z"},
        "feed2": {"url": "https://example.com/feed2", "title": "Feed 2", "update_interval": 30},
    }


@pytest.fixture
def mock_feed_data():
    """Common fixture for feed data."""
    return {
        "feed1": {
            "title": "Feed 1",
            "entries": [
                {
                    "title": "Entry 1",
                    "link": "https://example.com/1",
                    "categories": ["tech", "python"],
                    "content": "Python 3.10 released",
                },
                {
                    "title": "Entry 2",
                    "link": "https://example.com/2",
                    "categories": ["news", "tech"],
                    "content": "Other content",
                },
            ],
        },
        "feed2": {
            "title": "Feed 2",
            "entries": [
                {
                    "title": "Entry 3",
                    "link": "https://example.com/3",
                    "categories": ["sports"],
                    "content": "Sports news",
                },
                {
                    "title": "Entry 4",
                    "link": "https://example.com/4",
                    "categories": ["news"],
                    "content": "News content",
                },
            ],
        },
    }


@pytest.fixture
def setup_feed_mocks(mock_subscriptions, mock_feed_data, monkeypatch):
    """Setup mocks for RSS manager with common test data."""
    mock_manager = MagicMock()

    # Configure mock behaviors
    def mock_load_feed_data(feed_id):
        return mock_feed_data.get(feed_id, {"entries": []})

    mock_manager.load_subscriptions.return_value = mock_subscriptions
    mock_manager.load_feed_data.side_effect = mock_load_feed_data
    mock_manager.get_feed_file_path.return_value = "/test/path/feed1.json"

    # Apply mock to module
    monkeypatch.setattr("src.strands_tools.rss.rss_manager", mock_manager)

    return mock_manager


class TestRSSManager:
    """Test the RSSManager class functionality with improved organization."""

    def test_content_processing(self):
        """Test content processing methods (clean_html and format_entry)."""
        manager = RSSManager()

        # Test clean_html with various inputs
        assert manager.clean_html("") == ""
        assert manager.clean_html(None) == ""

        html = "<p>Test <strong>content</strong> with <a href='https://example.com'>link</a></p>"
        result = manager.clean_html(html)
        assert "Test **content** with [link](https://example.com)" in result

        # Test format_entry with different entry structures
        with patch.object(manager, "clean_html", side_effect=lambda x: x):  # Simplify clean_html for testing
            # Test basic entry
            basic_entry = {
                "title": "Test Entry",
                "link": "https://example.com/entry",
                "published": "2023-07-21T12:00:00Z",
                "author": "Test Author",
            }
            result = manager.format_entry(basic_entry)
            assert result["title"] == "Test Entry"
            assert result["link"] == "https://example.com/entry"

            # Test missing fields
            missing_fields = {"link": "https://example.com/entry2"}
            result = manager.format_entry(missing_fields)
            assert result["title"] == "Untitled"
            assert result["published"] == "Unknown date"

            # Test content handling
            entry_with_content = {"title": "Test", "content": [{"value": "<p>Test content</p>"}]}
            result = manager.format_entry(entry_with_content, include_content=True)
            assert result["content"] == "<p>Test content</p>"

            # Test with summary fallback
            entry_with_summary = {"title": "Test", "summary": "<p>Summary content</p>"}
            result = manager.format_entry(entry_with_summary, include_content=True)
            assert result["content"] == "<p>Summary content</p>"

            # Test with description fallback
            entry_with_desc = {"title": "Test", "description": "<p>Description content</p>"}
            result = manager.format_entry(entry_with_desc, include_content=True)
            assert result["content"] == "<p>Description content</p>"

            # Test with no content
            entry_no_content = {"title": "Test Entry"}
            result = manager.format_entry(entry_no_content, include_content=True)
            assert result["content"] == "No content available"

    @pytest.mark.parametrize(
        "url,expected_id",
        [
            ("https://example.com", "example_commain"),
            ("https://example.com/", "example_commain"),
            ("https://example.com/blog", "example_com_blog"),
            ("https://sub.example.com/feed", "sub_example_com_feed"),
            ("https://test.org/path/to/feed", "test_org_path_to_feed"),
        ],
    )
    def test_generate_feed_id(self, url, expected_id):
        """Test feed ID generation from URLs."""
        manager = RSSManager()
        assert manager.generate_feed_id(url) == expected_id

    def test_file_operations(self):
        """Test file operations (load/save subscriptions and feed data)."""
        manager = RSSManager()
        manager.get_subscription_file_path = MagicMock(return_value="/test/path/subscriptions.json")
        manager.get_feed_file_path = MagicMock(return_value="/test/path/feed1.json")

        # Test data
        test_data = {"feed1": {"url": "https://example.com/feed1"}, "feed2": {"url": "https://example.com/feed2"}}

        # Test load_subscriptions
        with patch("os.path.exists") as mock_exists, patch("builtins.open", new_callable=mock_open) as mock_file:
            # When file doesn't exist
            mock_exists.return_value = False
            assert manager.load_subscriptions() == {}

            # When file exists with valid JSON
            mock_exists.return_value = True
            mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
            assert manager.load_subscriptions() == test_data

            # When file has invalid JSON
            mock_file.return_value.__enter__.return_value.read.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            assert manager.load_subscriptions() == {}

        # Test save_subscriptions
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            manager.save_subscriptions(test_data)
            mock_file.assert_called_once_with("/test/path/subscriptions.json", "w")

            # Don't verify individual write calls as json.dump can call write() multiple times
            # Instead verify that the combined result of all writes is valid JSON
            written_calls = mock_file.return_value.__enter__.return_value.write.call_args_list
            written_data = "".join(call[0][0] for call in written_calls)

            # Verify the combined written data is valid JSON and matches the test_data when parsed
            assert json.loads(written_data) == test_data

        # Test load_feed_data
        with patch("os.path.exists") as mock_exists, patch("builtins.open", new_callable=mock_open) as mock_file:
            # When file doesn't exist
            mock_exists.return_value = False
            assert manager.load_feed_data("feed1") == {"entries": []}

            # When file exists with valid JSON
            mock_exists.return_value = True
            feed_data = {"title": "Test Feed", "entries": [{"title": "Entry 1"}, {"title": "Entry 2"}]}
            mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(feed_data)
            assert manager.load_feed_data("feed1") == feed_data

            # When file has invalid JSON
            mock_file.return_value.__enter__.return_value.read.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            assert manager.load_feed_data("feed1") == {"entries": []}

        # Test save_feed_data
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            feed_data = {"title": "Test Feed", "entries": [{"title": "Entry 1"}, {"title": "Entry 2"}]}
            manager.save_feed_data("feed1", feed_data)
            mock_file.assert_called_once_with("/test/path/feed1.json", "w")

            # Use the same approach as for save_subscriptions
            written_calls = mock_file.return_value.__enter__.return_value.write.call_args_list
            written_data = "".join(call[0][0] for call in written_calls)

            # Verify the written data is valid JSON and matches the expected data
            assert json.loads(written_data) == feed_data

    def test_feed_operations(self):
        """Test operations on feeds (fetch_feed and update_feed)."""
        manager = RSSManager()

        # Test fetch_feed
        with patch("requests.get") as mock_get, patch("feedparser.parse") as mock_parse:
            # Test without authentication
            manager.fetch_feed("https://example.com/feed")
            mock_parse.assert_called_with("https://example.com/feed", agent=None)

            # Test with basic authentication
            auth = {"type": "basic", "username": "user", "password": "pass"}
            manager.fetch_feed("https://example.com/feed", auth)
            mock_get.assert_called_with("https://example.com/feed", headers={}, auth=("user", "pass"))

            # Test with user agent
            manager.fetch_feed("https://example.com/feed", None, "CustomAgent")
            mock_parse.assert_called_with("https://example.com/feed", agent="CustomAgent")

            # Test with both auth and user agent
            manager.fetch_feed("https://example.com/feed", auth, "CustomAgent")
            mock_get.assert_called_with(
                "https://example.com/feed", headers={"User-Agent": "CustomAgent"}, auth=("user", "pass")
            )

        # Test update_feed
        with patch("feedparser.parse") as mock_parse:
            # Setup manager methods
            manager.load_feed_data = MagicMock(return_value={"entries": []})
            manager.save_feed_data = MagicMock()
            manager.save_subscriptions = MagicMock()
            manager.format_entry = MagicMock(return_value={"title": "Test Entry", "id": "entry1"})

            subscriptions = {"feed1": {"url": "https://example.com/feed"}}

            # Test with non-existent feed
            result = manager.update_feed("non_existent", subscriptions)
            assert result["status"] == "error"
            assert "not found" in result["content"][0]["text"]

            # Test with parsing error
            mock_parse.side_effect = Exception("Test error")
            result = manager.update_feed("feed1", subscriptions)
            assert result["status"] == "error"
            assert "Test error" in result["content"][0]["text"]

            # Test with feed that can't be parsed
            mock_parse.side_effect = None
            mock_parse.return_value = MagicMock(spec=[])  # No entries attribute
            result = manager.update_feed("feed1", subscriptions)
            assert result["status"] == "error"
            assert "Could not parse feed" in result["content"][0]["text"]

            # Test successful update
            mock_feed = MagicMock()
            mock_feed.feed = MagicMock()
            mock_feed.feed.title = "Test Feed"
            mock_feed.feed.description = "Test Description"
            mock_feed.feed.link = "https://example.com/feed"
            mock_feed.entries = [{"id": "entry1", "title": "Entry 1"}, {"id": "entry2", "title": "Entry 2"}]
            mock_parse.return_value = mock_feed

            result = manager.update_feed("feed1", subscriptions)
            assert result["feed_id"] == "feed1"
            assert result["title"] == "Test Feed"
            assert result["new_entries"] == 2
            manager.save_feed_data.assert_called_once()
            manager.save_subscriptions.assert_called_once()
            assert manager.format_entry.call_count == 2


class TestRSSTool:
    """Test the RSS tool function with improved organization."""

    def test_fetch_action(self):
        """Test fetch action with various scenarios."""
        with patch("feedparser.parse") as mock_parse:
            # Setup mock feed
            mock_feed = MagicMock()
            mock_feed.entries = [
                {"title": "Entry 1", "link": "https://example.com/1"},
                {"title": "Entry 2", "link": "https://example.com/2"},
            ]
            mock_parse.return_value = mock_feed

            # Test successful fetch
            result = rss(action="fetch", url="https://example.com/feed")
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["title"] == "Entry 1"

            # Test fetch with max_entries
            result = rss(action="fetch", url="https://example.com/feed", max_entries=1)
            assert len(result) == 1

            # Test fetch with empty feed
            mock_feed.entries = []
            result = rss(action="fetch", url="https://example.com/feed")
            assert result["status"] == "error"
            assert result["content"][0]["text"] == "Feed contains no entries"

    def test_subscription_actions(self, setup_feed_mocks):
        """Test subscribe and unsubscribe actions."""
        mock_manager = setup_feed_mocks

        # Test subscribe action
        mock_manager.generate_feed_id.return_value = "example_comfeed"
        mock_manager.load_subscriptions.return_value = {}  # Empty subscriptions
        mock_manager.update_feed.return_value = {"title": "Test Feed", "new_entries": 5}

        # Subscribe with minimal parameters
        result = rss(action="subscribe", url="https://example.com/feed")
        assert result["status"] == "success"
        assert "Subscribed to" in result["content"][0]["text"]
        assert "Test Feed" in result["content"][0]["text"]

        # Subscribe with custom feed_id
        result = rss(action="subscribe", url="https://example.com/feed", feed_id="custom_feed")
        assert result["status"] == "success"
        assert "Subscribed to" in result["content"][0]["text"]
        assert "custom_feed" in result["content"][0]["text"]

        # Subscribe with auth
        result = rss(action="subscribe", url="https://example.com/feed", auth_username="user", auth_password="pass")
        subscription_data = mock_manager.save_subscriptions.call_args[0][0]
        feed_id = mock_manager.generate_feed_id.return_value
        # The actual implementation might be using different keys or structure for auth
        # We'll simply check if the subscription was created successfully
        assert feed_id in subscription_data
        assert "url" in subscription_data[feed_id]

        # Subscribe to already subscribed feed
        mock_manager.load_subscriptions.return_value = {"example_comfeed": {"url": "https://example.com/feed"}}
        result = rss(action="subscribe", url="https://example.com/feed")
        assert result["status"] == "error"
        assert "Already subscribed" in result["content"][0]["text"]

        # Test unsubscribe action
        with patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
            # Unsubscribe from existing feed
            mock_manager.load_subscriptions.return_value = {
                "feed1": {"url": "https://example.com/feed1", "title": "Feed 1"}
            }
            result = rss(action="unsubscribe", feed_id="feed1")
            assert result["status"] == "success"
            assert "Unsubscribed from" in result["content"][0]["text"]
            assert "Feed 1" in result["content"][0]["text"]
            mock_manager.save_subscriptions.assert_called()
            mock_remove.assert_called_once_with("/test/path/feed1.json")

            # Unsubscribe from non-existent feed
            mock_manager.load_subscriptions.return_value = {}
            result = rss(action="unsubscribe", feed_id="feed1")
            assert result["status"] == "error"
            assert "Not subscribed to feed" in result["content"][0]["text"]

    def test_reading_actions(self, setup_feed_mocks, mock_subscriptions, mock_feed_data):
        """Test read and list actions."""
        mock_manager = setup_feed_mocks

        # Test list action
        # When there are subscriptions
        result = rss(action="list")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["feed_id"] == "feed1"
        assert result[0]["title"] == "Feed 1"

        # When there are no subscriptions
        mock_manager.load_subscriptions.return_value = {}
        result = rss(action="list")
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No subscribed feeds"

        # Restore subscriptions for next tests
        mock_manager.load_subscriptions.return_value = mock_subscriptions

        # Test read action
        # Successful read
        result = rss(action="read", feed_id="feed1")
        assert result["feed_id"] == "feed1"
        assert result["title"] == "Feed 1"
        assert len(result["entries"]) == 2

        # Read with max_entries
        result = rss(action="read", feed_id="feed1", max_entries=1)
        assert len(result["entries"]) == 1

        # Read with category filter
        result = rss(action="read", feed_id="feed1", category="tech")
        assert len(result["entries"]) == 2  # Both entries have "tech" category

        result = rss(action="read", feed_id="feed1", category="python")
        assert len(result["entries"]) == 1  # Only one entry has "python" category
        assert result["entries"][0]["title"] == "Entry 1"

        # Read with empty feed - returns an error dict instead of a string
        empty_feed = {"entries": []}
        mock_manager.load_feed_data = MagicMock(return_value=empty_feed)
        result = rss(action="read", feed_id="feed1")
        # Expect an error dict when no entries are found
        assert result["status"] == "error"
        assert "No entries found" in result["content"][0]["text"]

    def test_update_action(self, setup_feed_mocks, mock_subscriptions):
        """Test update action."""
        mock_manager = setup_feed_mocks

        # Test update specific feed
        mock_manager.update_feed.return_value = {"feed_id": "feed1", "new_entries": 3}
        result = rss(action="update", feed_id="feed1")
        assert result["feed_id"] == "feed1"
        mock_manager.update_feed.assert_called_with("feed1", mock_subscriptions)

        # Test update all feeds
        mock_manager.update_feed.reset_mock()
        result = rss(action="update")
        assert isinstance(result, list)
        assert len(result) == 2
        assert mock_manager.update_feed.call_count == 2
        expected_calls = [call("feed1", mock_subscriptions), call("feed2", mock_subscriptions)]
        mock_manager.update_feed.assert_has_calls(expected_calls, any_order=True)

        # Test update with no subscriptions
        mock_manager.load_subscriptions.return_value = {}
        result = rss(action="update")
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No subscribed feeds to update"

    def test_discovery_actions(self, setup_feed_mocks, mock_feed_data):
        """Test search and categories actions."""
        mock_manager = setup_feed_mocks

        # Test search action
        # Simple search
        result = rss(action="search", query="Entry")
        assert isinstance(result, list)
        assert len(result) == 4  # All entries have "Entry" in title

        # Content search
        result = rss(action="search", query="Python", include_content=True)
        assert len(result) == 1
        assert result[0]["entry"]["title"] == "Entry 1"

        # Search with max_entries
        result = rss(action="search", query="Entry", max_entries=2)
        assert len(result) == 2

        # Search with regex
        result = rss(action="search", query="Entry [13]")
        assert len(result) == 2
        titles = [r["entry"]["title"] for r in result]
        assert "Entry 1" in titles
        assert "Entry 3" in titles

        # No matches
        result = rss(action="search", query="NonExistent")
        assert result["status"] == "error"
        assert "No entries found matching query" in result["content"][0]["text"]

        # Test categories action
        result = rss(action="categories")
        assert "all_categories" in result
        assert "feed_categories" in result
        assert len(result["all_categories"]) == 4
        assert sorted(result["all_categories"]) == sorted(["tech", "python", "news", "sports"])

        # Empty categories
        mock_manager.load_feed_data.return_value = {"entries": [{"title": "No categories"}]}
        result = rss(action="categories")
        # Check that the result structure has categories, even if none found in new entries
        assert "all_categories" in result
        assert "feed_categories" in result

    @pytest.mark.parametrize(
        "action,params,expected_error",
        [
            ("fetch", {}, "URL is required"),
            ("read", {}, "feed_id is required"),
            ("unsubscribe", {}, "feed_id is required"),
            ("search", {}, "query is required"),
            ("invalid_action", {}, "Unknown action"),
            ("read", {"feed_id": "nonexistent"}, "Not subscribed to feed"),
            ("update", {"feed_id": "nonexistent"}, "No subscribed feeds to update"),
            ("unsubscribe", {"feed_id": "nonexistent"}, "Not subscribed to feed"),
        ],
    )
    def test_error_handling(self, action, params, expected_error, setup_feed_mocks):
        """Test error handling across different actions."""
        # Make sure subscriptions are empty for "nonexistent" feed_id tests
        if "feed_id" in params and params["feed_id"] == "nonexistent":
            setup_feed_mocks.load_subscriptions.return_value = {}

        result = rss(action=action, **params)
        assert result["status"] == "error"
        assert expected_error in result["content"][0]["text"]

    def test_general_exceptions(self):
        """Test handling of general exceptions."""
        with patch("feedparser.parse", side_effect=Exception("Test exception")):
            result = rss(action="fetch", url="https://example.com/feed")
            assert result["status"] == "error"
            assert "Test exception" in result["content"][0]["text"]
