import os
from unittest.mock import patch

from strands import Agent
from strands_tools import memory


def test_memory_store_and_list():
    """Test storing and listing documents in memory tool with mocks."""
    agent = Agent(tools=[memory])
    kb_id = "testkb123"

    # Set environment variables to bypass confirmation prompts and provide the KB ID.
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true", "STRANDS_KNOWLEDGE_BASE_ID": kb_id}):
        with (
            patch("strands_tools.memory.get_memory_service_client") as mock_get_client,
            patch("strands_tools.memory.get_memory_formatter") as mock_get_formatter,
        ):
            # Configure client mock
            mock_client = mock_get_client.return_value
            mock_client.get_data_source_id.return_value = "mock_ds_id"
            mock_client.store_document.return_value = ("mock_response", "x", "Test Title")
            mock_client.list_documents.return_value = {
                "documentDetails": [
                    {"identifier": {"custom": {"id": "test_doc"}}, "status": "ACTIVE", "updatedAt": "now"}
                ]
            }

            mock_formatter = mock_get_formatter.return_value
            mock_formatter.format_store_response.return_value = [{"text": "stored test_doc successfully"}]
            mock_formatter.format_list_response.return_value = [{"text": "listed test_doc successfully"}]

            # Store a document
            res_store = agent.tool.memory(action="store", content="integration test", title="Test Title")
            assert res_store["status"] == "success"
            assert "stored test_doc successfully" in str(res_store["content"])

            # List the documents to verify
            res_list = agent.tool.memory(action="list")
            assert res_list["status"] == "success"
            assert "listed test_doc successfully" in str(res_list["content"])
