"""
End-to-end test:
  • create / re-use a Bedrock KB
  • let Agent decide to call retrieve and bring it back
"""

import os
import time
from typing import Any, Generator
from unittest.mock import patch

import pytest
from strands import Agent
from strands_tools import memory, retrieve
from .utils.knowledge_base_util import KnowledgeBaseHelper

AWS_REGION = "us-east-1"
MIN_SCORE = "0.0"
WAIT_SECS = 10
MAX_POLLS = 12


@pytest.fixture(scope="module")
def kb_id() -> Generator[Any, Any, None]:
    helper = KnowledgeBaseHelper()
    kb_id = helper.try_get_existing() or helper.create_resources()
    yield kb_id
    if helper.should_teardown:
        helper.destroy()


@patch.dict(
    os.environ,
    {
        "BYPASS_TOOL_CONSENT": "true",
        "AWS_REGION": AWS_REGION,
        "MIN_SCORE": MIN_SCORE,
    },
    clear=False,
)
@pytest.mark.skip("KB retrieval takes longer in some cases, test is flaky")
def test_retrieve_semantic_search(kb_id):
    text = (
        "Python is a high-level programming language known for its simplicity. "
        "It supports object-oriented, functional and procedural paradigms."
    )
    ingest_agent = Agent(tools=[memory])
    store = ingest_agent.tool.memory(
        action="store",
        content=text,
        title="Python Guide",
        STRANDS_KNOWLEDGE_BASE_ID=kb_id,
        region_name=AWS_REGION,
    )
    assert store["status"] == "success", store
    doc_id = next(
        (c["text"].split(":", 1)[1].strip() for c in store["content"] if "Document ID" in c.get("text", "")),
        None,
    )
    assert doc_id, "Could not parse Document ID"

    bedrock_client = KnowledgeBaseHelper.get_boto_clients()["bedrock-agent"]
    ds = bedrock_client.list_data_sources(knowledgeBaseId=kb_id)["dataSourceSummaries"][0]["dataSourceId"]
    for _ in range(MAX_POLLS):
        details = bedrock_client.list_knowledge_base_documents(knowledgeBaseId=kb_id, dataSourceId=ds)[
            "documentDetails"
        ]
        if any(d["identifier"]["custom"]["id"] == doc_id and d["status"] == "INDEXED" for d in details):
            break
        time.sleep(WAIT_SECS)
    else:
        pytest.fail("Document never became INDEXED")

    retrieve_agent = Agent(tools=[retrieve])

    prompt = (
        f"Search knowledge base {kb_id} in region {AWS_REGION} for information about programming languages "
        f"and give me the result."
    )

    for _ in range(MAX_POLLS):
        reply = retrieve_agent(prompt)
        txt = str(reply).lower()
        if "python" in txt and "programming" in txt:
            return
        time.sleep(WAIT_SECS)

    pytest.fail("LLM never surfaced the stored document via retrieve")
