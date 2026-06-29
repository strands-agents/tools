"""
End-to-end test for the mem0_memory tool against the real mem0 (>=2.0) library.

This exercises the OSS ``Memory`` code path (local FAISS vector store) where the
mem0 2.x breaking change lives: ``get_all``/``search`` now require entity ids to
be passed via ``filters=`` instead of as top-level keyword arguments. A mocked
test cannot catch that regression, so this runs the real store -> search -> list
-> get -> history -> delete cycle.

Backend: OpenAI embedder + LLM (so it runs locally without AWS). The OpenAI
embedder is pinned to ``embedding_dims=1024`` to match the tool's hardcoded FAISS
index dimension. The test skips itself when ``OPENAI_API_KEY`` is not set.

Requires ``faiss-cpu`` to be installed (the ``mem0-memory`` extra does not pull
it). Run via: ``hatch run test-integ tests_integ/test_mem0_memory_e2e.py``
"""

import glob
import os
import uuid
from typing import Any, Dict, List

import pytest

from strands_tools.mem0_memory import Mem0ServiceClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping real mem0 2.x e2e",
)

# Matches the path hardcoded in Mem0ServiceClient._append_faiss_config.
_FAISS_PATH = "/tmp/mem0_384_faiss"

OPENAI_CONFIG = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "embedding_dims": 1024},
    },
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
}


def _results(resp: Any) -> List[Dict[str, Any]]:
    """Normalize a mem0 response into a list of memory dicts."""
    if isinstance(resp, list):
        return resp
    return resp.get("results", [])


@pytest.fixture
def fresh_faiss_store():
    """Remove any stale local FAISS index so the 1024-dim index is created clean."""
    for path in glob.glob(f"{_FAISS_PATH}*"):
        try:
            os.remove(path)
        except (IsADirectoryError, OSError):
            import shutil

            shutil.rmtree(path, ignore_errors=True)
    yield


@pytest.fixture
def client(fresh_faiss_store):
    # No MEM0_API_KEY / OPENSEARCH_HOST / NEPTUNE_* env vars -> default FAISS path.
    return Mem0ServiceClient(config=OPENAI_CONFIG)


def test_mem0_memory_full_cycle(client):
    """store -> search -> list -> get -> history -> delete against real mem0 2.x."""
    user_id = f"e2e_user_{uuid.uuid4().hex[:8]}"
    fact = "My name is Alex and I love hiking in the Swiss Alps every summer."

    # store
    store_resp = client.store_memory(content=fact, user_id=user_id)
    assert _results(store_resp), "store should create at least one memory"

    # search -- this is the call that raised the 2.x ValueError before the
    # filters= fix. Assert the stored fact is actually retrievable.
    search_resp = client.search_memories(query="What are my outdoor hobbies?", user_id=user_id)
    search_results = _results(search_resp)
    assert search_results, "search should return the stored memory"
    assert any("hik" in (m.get("memory", "").lower()) for m in search_results), (
        f"expected a hiking-related memory, got: {[m.get('memory') for m in search_results]}"
    )

    # list -- also goes through get_all(filters=...)
    list_results = _results(client.list_memories(user_id=user_id))
    assert list_results, "list should return the stored memory"
    memory_id = list_results[0]["id"]

    # get
    got = client.get_memory(memory_id)
    assert got.get("id") == memory_id

    # history
    history = client.get_memory_history(memory_id)
    assert isinstance(history, list)

    # delete -> confirm it is gone
    client.delete_memory(memory_id)
    remaining_ids = [m["id"] for m in _results(client.list_memories(user_id=user_id))]
    assert memory_id not in remaining_ids
