"""
Tests for browser models with actual logic methods.
"""

from strands_tools.browser.models import BrowserSession


def test_browser_session_tab_management():
    """Test BrowserSession tab management logic."""
    session = BrowserSession(
        session_name="test-session",
        description="Test session description",
    )

    # Initial state
    assert session.tabs == {}
    assert session.active_tab_id is None

    # Test adding tabs
    session.add_tab("tab1", "mock-page-1")
    assert "tab1" in session.tabs
    assert session.active_tab_id == "tab1"

    session.add_tab("tab2", "mock-page-2")
    assert len(session.tabs) == 2
    assert session.active_tab_id == "tab2"  # Should switch the new tab

    # Test tab switching
    assert session.switch_tab("tab1") is True
    assert session.active_tab_id == "tab1"
    assert session.switch_tab("nonexistent") is False

    # Test tab removal
    assert session.remove_tab("tab1") is True
    assert "tab1" not in session.tabs
    assert session.active_tab_id == "tab2"  # Should switch to remaining tab

    assert session.remove_tab("tab2") is True
    assert len(session.tabs) == 0
    assert session.active_tab_id is None


def test_browser_session_get_active_page():
    """Test getting the active page from a session."""
    session = BrowserSession(session_name="active-test", description="Active page test")

    # Test with no tabs
    assert session.get_active_page() is None

    # Test with tabs
    mock_page1 = "mock-page-1"
    mock_page2 = "mock-page-2"
    session.add_tab("tab1", mock_page1)
    session.add_tab("tab2", mock_page2)

    # Should return active tab's page
    assert session.get_active_page() == mock_page2

    # Switch and test again
    session.switch_tab("tab1")
    assert session.get_active_page() == mock_page1
