"""
Integration test for the Bedrock Knowledge Base (memory) tool.

This test creates real AWS resources (IAM Role, OpenSearch Collection, Bedrock KB)
to validate the end-to-end functionality of the memory tool.
"""

import os
import time
import uuid
from unittest.mock import patch

import pytest
from strands import Agent
from strands_tools import memory
from .utils.knowledge_base_util import KnowledgeBaseHelper

AWS_REGION = "us-east-1"


@pytest.fixture(scope="module")
def managed_knowledge_base():
    helper = KnowledgeBaseHelper()
    kb_id = helper.try_get_existing()
    if kb_id is not None:
        yield kb_id
    else:
        kb_id = helper.create_resources()
        yield kb_id
        if helper.should_teardown:
            helper.destroy()

@pytest.mark.skip("memory ingestion takes longer in some cases, test is flaky")
@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_memory_integration_store_and_retrieve(managed_knowledge_base):
    """
    End-to-end test for Bedrock Knowledge Base memory tool:
      - Store a unique document
      - Poll until it is INDEXED
      - Retrieve via semantic search and verify presence
    """
    kb_id = managed_knowledge_base
    agent = Agent(tools=[memory])
    clients = KnowledgeBaseHelper.get_boto_clients()

    test_uuid = str(uuid.uuid4())
    unique_content = f"The secret password for the test is {test_uuid}."
    store_result = agent.tool.memory(
        action="store",
        content=unique_content,
        title="Integration Test Document",
        STRANDS_KNOWLEDGE_BASE_ID=kb_id,
        region_name=AWS_REGION,
    )
    assert store_result["status"] == "success", f"Store failed: {store_result}"

    # Extract document ID
    doc_id = next(
        (
            item["text"].split(":", 1)[1].strip()
            for item in store_result["content"]
            if "Document ID" in item.get("text", "")
        ),
        None,
    )
    assert doc_id, f"No document_id returned from store operation. Got: {store_result}"

    ds_id = clients["bedrock-agent"].list_data_sources(knowledgeBaseId=kb_id)["dataSourceSummaries"][0]["dataSourceId"]
    for _ in range(18):
        docs = clients["bedrock-agent"].list_knowledge_base_documents(knowledgeBaseId=kb_id, dataSourceId=ds_id)[
            "documentDetails"
        ]
        found = next((d for d in docs if d.get("identifier", {}).get("custom", {}).get("id") == doc_id), None)
        if found and found.get("status") == "INDEXED":
            break
        time.sleep(10)
    else:
        raise AssertionError("Stored document did not become INDEXED in time.")

    # Try up to 2 minutes for the content to appear in retrieval results
    for _ in range(12):
        retrieve_result = agent.tool.memory(
            action="retrieve",
            query=test_uuid,  # use uuid as query is always order by semantic relevance
            STRANDS_KNOWLEDGE_BASE_ID=kb_id,
            region_name=AWS_REGION,
            min_score=0.0,
            max_results=10,
        )

        full_retrieved_text = " ".join(item.get("text", "") for item in retrieve_result.get("content", []))
        if unique_content in full_retrieved_text:
            break
        time.sleep(10)
    else:
        raise AssertionError("Stored content not found in retrieval after waiting.")
