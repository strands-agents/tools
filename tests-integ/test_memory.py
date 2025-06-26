import os
from unittest.mock import MagicMock, patch

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


def test_memory_tool_should_adapt_to_existing_datasource_type_should_fail_now():
    """
    This test is designed to FAIL at the final assertion until the tool bug is fixed.

    It verifies that the memory tool correctly detects and uses the 'S3' dataSourceType
    from an existing Knowledge Base, rather than hardcoding 'CUSTOM'.
    """
    agent = Agent(tools=[memory])
    kb_id = "S3KNOWLEDGEBASEID"

    with patch("strands_tools.memory.boto3.Session.client") as mock_boto_client:
        # 1. Configure the mock that `boto3.Session.client` will return.
        mock_agent_client = MagicMock()
        mock_boto_client.return_value = mock_agent_client

        # 2. Simulate the response for `list_data_sources`. This is the first
        #    API call the tool makes. We make it return a summary indicating
        #    the data source type is 'S3'.
        mock_agent_client.list_data_sources.return_value = {
            "dataSourceSummaries": [{"dataSourceId": "mock_s3_datasource", "type": "S3"}]
        }

        # 3. Mock the final ingestion call to prevent a real API call.
        mock_agent_client.ingest_knowledge_base_documents.return_value = {}

        # Set environment variables to bypass confirmation prompts.
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true", "STRANDS_KNOWLEDGE_BASE_ID": kb_id}):
            agent.tool.memory(
                action="store", content="This content should be stored via S3.", title="S3 Adaptation Test"
            )

            # --- Assertions ---
            # This is the key assertion that is INTENDED TO FAIL until the bug is fixed.
            # It checks that the tool's logic correctly constructed the API request
            # with 'S3' as the dataSourceType, which it should have discovered.

            # First, ensure the final API call was even made.
            mock_agent_client.ingest_knowledge_base_documents.assert_called_once()

            # Then, inspect the keyword arguments of that call.
            call_kwargs = mock_agent_client.ingest_knowledge_base_documents.call_args.kwargs
            sent_document_payload = call_kwargs["documents"][0]

            # This is the line that will fail now, but will pass when the tool is fixed.
            assert sent_document_payload["content"]["dataSourceType"] == "S3"
