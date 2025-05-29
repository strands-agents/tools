"""
Tests for the retrieve tool using the Agent interface.
"""

import os
from unittest import mock

import boto3
import pytest
from strands import Agent
from strands_tools import retrieve


@pytest.fixture
def agent():
    """Create an agent with the retrieve tool loaded."""
    return Agent(tools=[retrieve])


@pytest.fixture
def mock_boto3_client():
    """Mock the boto3 client to avoid actual AWS calls during tests."""
    with mock.patch.object(boto3, "client") as mock_client:
        # Create a mock response object
        mock_response = {
            "retrievalResults": [
                {
                    "content": {"text": "Test content 1", "type": "TEXT"},
                    "location": {
                        "customDocumentLocation": {"id": "doc-001"},
                        "type": "CUSTOM",
                    },
                    "metadata": {"source": "test-source-1"},
                    "score": 0.9,
                },
                {
                    "content": {"text": "Test content 2", "type": "TEXT"},
                    "location": {
                        "customDocumentLocation": {"id": "doc-002"},
                        "type": "CUSTOM",
                    },
                    "metadata": {"source": "test-source-2"},
                    "score": 0.7,
                },
                {
                    "content": {"text": "Test content 3", "type": "TEXT"},
                    "location": {
                        "customDocumentLocation": {"id": "doc-003"},
                        "type": "CUSTOM",
                    },
                    "metadata": {"source": "test-source-3"},
                    "score": 0.3,
                },
            ]
        }

        # Configure the mock client to return our mock response
        mock_client_instance = mock_client.return_value
        mock_client_instance.retrieve.return_value = mock_response

        yield mock_client


@pytest.fixture(autouse=True)
def os_environment():
    mock_env = {}
    with mock.patch.object(os, "environ", mock_env):
        yield mock_env


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_filter_results_by_score():
    """Test the filter_results_by_score function."""
    test_results = [{"score": 0.9}, {"score": 0.5}, {"score": 0.3}, {"score": 0.8}]

    # Filter with threshold 0.5
    filtered = retrieve.filter_results_by_score(test_results, 0.5)
    assert len(filtered) == 3
    assert filtered[0]["score"] == 0.9
    assert filtered[1]["score"] == 0.5
    assert filtered[2]["score"] == 0.8

    # Filter with threshold 0.8
    filtered = retrieve.filter_results_by_score(test_results, 0.8)
    assert len(filtered) == 2
    assert filtered[0]["score"] == 0.9
    assert filtered[1]["score"] == 0.8


def test_format_results_for_display():
    """Test the format_results_for_display function."""
    test_results = [
        {
            "content": {"text": "Sample content", "type": "TEXT"},
            "location": {
                "customDocumentLocation": {"id": "test-doc-1"},
                "type": "CUSTOM",
            },
            "score": 0.95,
        }
    ]

    formatted = retrieve.format_results_for_display(test_results)
    assert "Score: 0.9500" in formatted
    assert "Document ID: test-doc-1" in formatted
    assert "Content: Sample content" in formatted

    # Test with empty results
    empty_formatted = retrieve.format_results_for_display([])
    assert empty_formatted == "No results found above score threshold."


def test_retrieve_tool_direct(mock_boto3_client):
    """Test direct invocation of the retrieve tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "text": "test query",
            "knowledgeBaseId": "test-kb-id",
            "numberOfResults": 3,
        },
    }

    # Call the retrieve function directly
    with mock.patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": "default-kb-id"}):
        result = retrieve.retrieve(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Retrieved 2 results with score >= 0.4" in result["content"][0]["text"]

    # Verify that boto3 client was called with correct parameters
    mock_boto3_client.assert_called_once_with("bedrock-agent-runtime", region_name="us-west-2")
    mock_boto3_client.return_value.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="test-kb-id",
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
    )


def test_retrieve_with_default_kb_id(mock_boto3_client):
    """Test retrieve tool using default knowledge base ID from environment."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"text": "test query"}}

    # Set environment variable for knowledge base ID
    with mock.patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": "default-kb-id"}):
        result = retrieve.retrieve(tool=tool_use)

    # Verify that boto3 client was called with the default KB ID
    mock_boto3_client.return_value.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="default-kb-id",
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

    assert result["status"] == "success"


def test_retrieve_error_handling(mock_boto3_client):
    """Test error handling in the retrieve tool."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "text": "test query",
        },
    }

    # Configure mock to raise an exception
    mock_boto3_client.return_value.retrieve.side_effect = Exception("Test error")

    result = retrieve.retrieve(tool=tool_use)

    # Verify the error result
    assert result["status"] == "error"
    assert "Error during retrieval: Test error" in result["content"][0]["text"]


def test_retrieve_custom_score_threshold(mock_boto3_client):
    """Test retrieve with custom score threshold."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "text": "test query",
            "knowledgeBaseId": "test-kb-id",
            "score": 0.8,  # Higher threshold than default
        },
    }

    result = retrieve.retrieve(tool=tool_use)

    # Should only get results with score >= 0.8
    assert result["status"] == "success"
    assert "Retrieved 1 results with score >= 0.8" in result["content"][0]["text"]
    # Only the highest score result (0.9) should be included
    assert "Score: 0.9" in result["content"][0]["text"]
    assert "doc-001" in result["content"][0]["text"]
    # Medium score result (0.7) should not be included
    assert "doc-002" not in result["content"][0]["text"]


def test_retrieve_via_agent(agent, mock_boto3_client):
    """Test retrieving via the agent interface."""
    with mock.patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": "agent-kb-id"}):
        result = agent.tool.retrieve(text="agent query", knowledgeBaseId="test-kb-id")

    result_text = extract_result_text(result)
    assert "Retrieved" in result_text
    assert "results with score >=" in result_text

    # Verify the boto3 client was called with correct parameters
    mock_boto3_client.return_value.retrieve.assert_called_once_with(
        retrievalQuery={"text": "agent query"},
        knowledgeBaseId="test-kb-id",
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )


def test_retrieve_with_custom_profile(mock_boto3_client):
    """Test retrieve with custom AWS profile."""
    with mock.patch.object(boto3, "Session") as mock_session:
        # Configure mock session
        mock_session_instance = mock_session.return_value
        mock_session_instance.client.return_value = mock_boto3_client.return_value

        # Call retrieve with custom profile
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {"text": "test query", "profile_name": "custom-profile"},
        }

        result = retrieve.retrieve(tool=tool_use)

        # Verify session was created with correct profile
        mock_session.assert_called_once_with(profile_name="custom-profile")
        mock_session_instance.client.assert_called_once_with("bedrock-agent-runtime", region_name="us-west-2")

        # Verify result
        assert result["status"] == "success"


def test_retrieve_with_custom_region():
    """Test retrieve with custom AWS region."""
    with mock.patch.object(boto3, "client") as mock_client:
        # Configure mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Custom region content", "type": "TEXT"},
                    "location": {
                        "customDocumentLocation": {"id": "doc-region"},
                        "type": "CUSTOM",
                    },
                    "score": 0.85,
                }
            ]
        }

        # Call retrieve with custom region
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "text": "test query",
                "region": "us-east-1",
                "knowledgeBaseId": "region-kb-id",
            },
        }

        result = retrieve.retrieve(tool=tool_use)

        # Verify client was created with correct region
        mock_client.assert_called_once_with("bedrock-agent-runtime", region_name="us-east-1")

        # Verify result
        assert result["status"] == "success"
        assert "Retrieved 1 results" in result["content"][0]["text"]
        assert "Custom region content" in result["content"][0]["text"]


def test_retrieve_no_results_above_threshold(mock_boto3_client):
    """Test retrieve when no results are above the threshold."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "text": "test query",
            "score": 0.95,  # Higher than any result in our mock data
        },
    }

    result = retrieve.retrieve(tool=tool_use)

    # Verify the result shows no items above threshold
    assert result["status"] == "success"
    assert "Retrieved 0 results with score >= 0.95" in result["content"][0]["text"]
    assert "No results found above score threshold" in result["content"][0]["text"]


def test_format_results_non_string_content():
    """Test format_results_for_display with non-string content."""
    # Test case where content["text"] is not a string
    test_results = [
        {
            "content": {"text": 12345, "type": "TEXT"},  # Non-string text
            "location": {
                "customDocumentLocation": {"id": "test-doc-1"},
                "type": "CUSTOM",
            },
            "score": 0.95,
        }
    ]

    formatted = retrieve.format_results_for_display(test_results)
    assert "Score: 0.9500" in formatted
    assert "Document ID: test-doc-1" in formatted
    # Content should not be included since text is not a string
    assert "Content:" not in formatted
