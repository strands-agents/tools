"""Tests for the apply_patch tool."""

import pytest
from strands import Agent

from strands_tools import apply_patch
from strands_tools.apply_patch import anchor_hash


@pytest.fixture
def agent():
    return Agent(tools=[apply_patch])


@pytest.fixture
def file_with_text(tmp_path):
    path = tmp_path / "file.py"
    path.write_text("alpha\nbeta\ngamma\n")
    return str(path)


def _payload(tool_response):
    for block in tool_response["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError("No JSON content block in tool response")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_single_patch_applies(agent, file_with_text):
    old = "beta\n"
    response = agent.tool.apply_patch(
        path=file_with_text,
        patches=[{"anchor_hash": anchor_hash(old), "old": old, "new": "BETA\n"}],
    )

    assert response["status"] == "success"
    assert _payload(response)["applied"] == 1
    assert _read(file_with_text) == "alpha\nBETA\ngamma\n"


def test_multiple_patches_applied_in_order(agent, file_with_text):
    p1_old, p1_new = "alpha\n", "ALPHA\n"
    p2_old, p2_new = "gamma\n", "GAMMA\n"

    response = agent.tool.apply_patch(
        path=file_with_text,
        patches=[
            {"anchor_hash": anchor_hash(p1_old), "old": p1_old, "new": p1_new},
            {"anchor_hash": anchor_hash(p2_old), "old": p2_old, "new": p2_new},
        ],
    )

    assert response["status"] == "success"
    assert _payload(response)["applied"] == 2
    assert _read(file_with_text) == "ALPHA\nbeta\nGAMMA\n"


def test_anchor_mismatch_leaves_file_untouched(agent, file_with_text):
    response = agent.tool.apply_patch(
        path=file_with_text,
        patches=[{"anchor_hash": "deadbeef", "old": "beta\n", "new": "BETA\n"}],
    )

    assert response["status"] == "error"
    assert _payload(response)["results"][0]["status"] == "anchor_mismatch"
    assert _read(file_with_text) == "alpha\nbeta\ngamma\n"


def test_not_found_leaves_file_untouched(agent, file_with_text):
    missing = "delta\n"
    response = agent.tool.apply_patch(
        path=file_with_text,
        patches=[{"anchor_hash": anchor_hash(missing), "old": missing, "new": "DELTA\n"}],
    )

    assert response["status"] == "error"
    assert _payload(response)["results"][0]["status"] == "not_found"
    assert _read(file_with_text) == "alpha\nbeta\ngamma\n"


def test_ambiguous_match_leaves_file_untouched(agent, tmp_path):
    path = tmp_path / "f.py"
    path.write_text("x\nx\ny\n")

    response = agent.tool.apply_patch(
        path=str(path),
        patches=[{"anchor_hash": anchor_hash("x\n"), "old": "x\n", "new": "X\n"}],
    )

    assert response["status"] == "error"
    payload = _payload(response)
    assert payload["results"][0]["status"] == "ambiguous"
    assert payload["results"][0]["matches"] == 2
    assert _read(str(path)) == "x\nx\ny\n"


def test_failure_in_second_patch_aborts_all(agent, file_with_text):
    good_old, good_new = "alpha\n", "ALPHA\n"
    bad_old = "missing\n"

    response = agent.tool.apply_patch(
        path=file_with_text,
        patches=[
            {"anchor_hash": anchor_hash(good_old), "old": good_old, "new": good_new},
            {"anchor_hash": anchor_hash(bad_old), "old": bad_old, "new": "X\n"},
        ],
    )

    assert response["status"] == "error"
    payload = _payload(response)
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][1]["status"] == "not_found"
    assert _read(file_with_text) == "alpha\nbeta\ngamma\n"


def test_empty_patches_is_an_error(agent, file_with_text):
    response = agent.tool.apply_patch(path=file_with_text, patches=[])
    assert response["status"] == "error"
    assert "non-empty" in _payload(response)["error"]


def test_missing_file_returns_error(agent, tmp_path):
    response = agent.tool.apply_patch(
        path=str(tmp_path / "nope.py"),
        patches=[{"anchor_hash": anchor_hash("x"), "old": "x", "new": "y"}],
    )
    assert response["status"] == "error"
    assert "Could not read" in _payload(response)["error"]
