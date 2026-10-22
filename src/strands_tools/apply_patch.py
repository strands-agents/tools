"""Hash-anchored file edits.

Apply one or more edits to a file by anchoring each edit on a short content
hash of the lines being replaced. If the file shifts in unrelated places, the
hash for the targeted span still matches and the edit lands. If the targeted
span itself has changed, the hash mismatches and the edit fails cleanly
instead of corrupting the file.

This is a port of the idea from the Dirac TerminalBench writeup ("hash anchors
+ Myers diff in a single token"). The hash is short, deterministic, and easy
for an LLM to emit.

Each patch entry is:
    {
        "anchor_hash": "a1b2c3d4",         # 8-hex-char prefix of sha256 of `old`
        "old": "    return value\n",
        "new": "    return value + 1\n",
    }

Resolution:
    1. Compute sha256 of `old` and confirm its 8-char prefix matches anchor_hash.
       This catches transcription errors before touching the file.
    2. Search for `old` as a substring in the current file content.
    3. If exactly one match exists, replace it with `new`. Otherwise the patch
       entry fails: zero matches means the file changed, multiple matches
       means the anchor is ambiguous and `old` needs more context.

Patches are applied in order. If any entry fails, the file is left untouched
(write is atomic via a tempfile rename). The result lists per-entry status so
the agent can decide which entries to fix and retry.

Usage:
    from strands import Agent
    from strands_tools import apply_patch

    agent = Agent(tools=[apply_patch])
    agent.tool.apply_patch(
        path="/repo/src/module.py",
        patches=[
            {
                "anchor_hash": "a1b2c3d4",
                "old": "x = 1\n",
                "new": "x = 2\n",
            },
        ],
    )
"""

import hashlib
import os
import tempfile
from os.path import expanduser
from typing import Any, Dict, List

from strands import tool

ANCHOR_LENGTH = 8


def anchor_hash(text: str) -> str:
    """Compute the short hex anchor for a span of text.

    Public so callers can pre-compute anchors when crafting patches without
    invoking the tool.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ANCHOR_LENGTH]


def _apply_one(content: str, patch: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        return {"index": index, "status": "error", "error": "Patch entry must be an object."}

    old = patch.get("old")
    new = patch.get("new")
    declared = patch.get("anchor_hash")

    if not isinstance(old, str) or not isinstance(new, str):
        return {
            "index": index,
            "status": "error",
            "error": "Patch entry requires string fields 'old' and 'new'.",
        }
    if not isinstance(declared, str):
        return {
            "index": index,
            "status": "error",
            "error": "Patch entry requires a string 'anchor_hash'.",
        }

    expected = anchor_hash(old)
    if declared.lower() != expected:
        return {
            "index": index,
            "status": "anchor_mismatch",
            "expected_anchor": expected,
            "received_anchor": declared,
            "error": ("anchor_hash does not match sha256(old)[:8]. The 'old' text was likely transcribed incorrectly."),
        }

    occurrences = content.count(old)
    if occurrences == 0:
        return {
            "index": index,
            "status": "not_found",
            "error": "'old' was not found in the file. The targeted region may have changed.",
        }
    if occurrences > 1:
        return {
            "index": index,
            "status": "ambiguous",
            "matches": occurrences,
            "error": "'old' matched more than one location. Add surrounding context to disambiguate.",
        }

    return {
        "index": index,
        "status": "success",
        "anchor_hash": expected,
        "applied_content": content.replace(old, new, 1),
    }


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".apply_patch_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@tool
def apply_patch(path: str, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply one or more hash-anchored edits to a file.

    Each patch carries an `anchor_hash` (the 8-character sha256 prefix of its
    `old` text), an `old` span to replace, and the `new` span to substitute.
    The edit lands only if the anchor verifies and `old` appears exactly once
    in the current file.

    All patches must succeed for the file to be written. On any failure the
    file is left unchanged and the failing entries are reported in the result.

    Args:
        path: Absolute or user-relative path to the file to edit. Must exist.
        patches: Ordered list of patch entries. Each entry must have keys
            `anchor_hash`, `old`, and `new`, all strings.

    Returns:
        ToolResult dict. The JSON content block carries:
            - path: The expanded path.
            - applied: count of entries that applied.
            - failed: count of entries that did not.
            - results: per-entry detail. Failed entries carry status values
              "anchor_mismatch", "not_found", "ambiguous", or "error".
            - error: present only on overall failure.

    Examples:
        >>> apply_patch(
        ...     path="/repo/file.py",
        ...     patches=[
        ...         {"anchor_hash": "a1b2c3d4", "old": "x=1\\n", "new": "x=2\\n"},
        ...     ],
        ... )
    """
    expanded = expanduser(path)

    def _wrap(payload: Dict[str, Any], status: str) -> Dict[str, Any]:
        applied = payload.get("applied", 0)
        failed = payload.get("failed", 0)
        if "error" in payload:
            text = f"apply_patch: {payload['error']}"
        else:
            text = f"apply_patch: applied {applied}, failed {failed} on {expanded}"
        return {
            "status": status,
            "content": [{"text": text}, {"json": payload}],
        }

    if not isinstance(patches, list) or not patches:
        return _wrap(
            {
                "path": expanded,
                "error": "'patches' must be a non-empty list.",
                "applied": 0,
                "failed": 0,
                "results": [],
            },
            "error",
        )

    try:
        with open(expanded, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        return _wrap(
            {
                "path": expanded,
                "error": f"Could not read file: {exc}",
                "applied": 0,
                "failed": len(patches),
                "results": [],
            },
            "error",
        )

    results: List[Dict[str, Any]] = []
    working = content

    for index, patch in enumerate(patches):
        outcome = _apply_one(working, patch, index)
        if outcome["status"] == "success":
            working = outcome.pop("applied_content")
        results.append(outcome)

    failed = [r for r in results if r["status"] != "success"]
    if failed:
        return _wrap(
            {
                "path": expanded,
                "applied": 0,
                "failed": len(failed),
                "results": results,
                "error": "One or more patch entries failed; file was not modified.",
            },
            "error",
        )

    try:
        _atomic_write(expanded, working)
    except OSError as exc:
        return _wrap(
            {
                "path": expanded,
                "error": f"Could not write file: {exc}",
                "applied": 0,
                "failed": len(patches),
                "results": results,
            },
            "error",
        )

    return _wrap(
        {
            "path": expanded,
            "applied": len(results),
            "failed": 0,
            "results": results,
        },
        "success",
    )
