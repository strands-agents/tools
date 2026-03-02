"""
Tests for the skills tool.

Tests cover:
- Skill listing
- Loading skill instructions
- Resource loading
- Error handling
- get_skills_prompt helper
- Import action (directory, file, URL)
- Virtual skill behavior
- Experimental warning
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import skills


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    # Create code-reviewer skill
    code_reviewer = tmp_path / "code-reviewer"
    code_reviewer.mkdir()
    (code_reviewer / "SKILL.md").write_text("""---
name: code-reviewer
description: Performs thorough code reviews with focus on security and best practices.
license: Apache-2.0
---

# Code Reviewer Skill

## Overview

This skill enables thorough, consistent code reviews.

## Guidelines

- Check for security vulnerabilities
- Review code style
- Verify error handling
""")

    # Create script resource
    scripts_dir = code_reviewer / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze.py").write_text("""#!/usr/bin/env python3
# Code analysis script
def analyze(code):
    return {"issues": []}
""")

    # Create references
    refs_dir = code_reviewer / "references"
    refs_dir.mkdir()
    (refs_dir / "security.md").write_text("# Security Guidelines\n\n- Always sanitize input\n")

    # Create data-analyst skill
    data_analyst = tmp_path / "data-analyst"
    data_analyst.mkdir()
    (data_analyst / "SKILL.md").write_text("""---
name: data-analyst
description: Analyzes data and provides insights using pandas and visualization.
---

# Data Analyst Skill

## Overview

This skill enables systematic data analysis.

## Workflow

1. Load data
2. Clean data
3. Analyze patterns
4. Visualize results
""")

    # Create skill with invalid name (should be skipped)
    invalid_skill = tmp_path / "Invalid_Skill"
    invalid_skill.mkdir()
    (invalid_skill / "SKILL.md").write_text("""---
name: Invalid_Skill
description: This should be skipped due to invalid name.
---

# Invalid
""")

    return tmp_path


@pytest.fixture
def agent():
    """Create an agent with the skills tool loaded."""
    return Agent(tools=[skills])


def extract_text(result):
    """Extract text from tool result."""
    if isinstance(result, dict):
        if "content" in result and isinstance(result["content"], list):
            return result["content"][0].get("text", "")
    return str(result)


class TestSkillListing:
    """Tests for skill listing functionality."""

    def test_list_skills(self, agent, skills_dir):
        """Test listing skills in a directory."""
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "code-reviewer" in text
        assert "data-analyst" in text
        assert "Invalid_Skill" not in text
        assert "Available skills (2)" in text

    def test_list_empty_directory(self, agent, tmp_path):
        """Test listing skills in an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(empty_dir))
        text = extract_text(result)

        assert "No skills found" in text

    def test_list_nonexistent_directory(self, agent, tmp_path):
        """Test listing skills in a nonexistent directory."""
        nonexistent = str(tmp_path / "nonexistent")
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=nonexistent)
        text = extract_text(result)

        assert "No skills found" in text


class TestSkillUsage:
    """Tests for loading skill instructions."""

    def test_use_skill(self, agent, skills_dir):
        """Test loading a skill's instructions."""
        result = agent.tool.skills(action="use", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Skill: code-reviewer" in text
        assert "Guidelines" in text
        assert "security vulnerabilities" in text.lower()

    def test_use_nonexistent_skill(self, agent, skills_dir):
        """Test loading a skill that doesn't exist."""
        result = agent.tool.skills(action="use", skill_name="nonexistent", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower()

    def test_use_skill_missing_name(self, agent, skills_dir):
        """Test error when skill_name is not provided."""
        result = agent.tool.skills(action="use", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "skill_name is required" in text


class TestSkillResources:
    """Tests for skill resource handling."""

    def test_list_resources(self, agent, skills_dir):
        """Test listing skill resources."""
        result = agent.tool.skills(
            action="list_resources",
            skill_name="code-reviewer",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert "scripts/" in text
        assert "analyze.py" in text
        assert "references/" in text
        assert "security.md" in text

    def test_get_resource(self, agent, skills_dir):
        """Test loading a resource file."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="scripts/analyze.py",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "def analyze" in text

    def test_get_nonexistent_resource(self, agent, skills_dir):
        """Test loading a resource that doesn't exist."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="nonexistent.py",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"

    def test_get_resource_path_traversal(self, agent, skills_dir):
        """Test that path traversal is blocked."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="../data-analyst/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_action(self, agent, skills_dir):
        """Test handling of invalid action."""
        result = agent.tool.skills(action="invalid_action", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "error" in text.lower() or "invalid" in text.lower()

    def test_missing_resource_path(self, agent, skills_dir):
        """Test error when resource_path is required but not provided."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "resource_path is required" in text


class TestAutoDiscovery:
    """Tests for auto-discovery feature."""

    def test_auto_discover_from_env(self, agent, skills_dir, monkeypatch):
        """Test that skills are auto-discovered from env var."""
        monkeypatch.setenv("STRANDS_SKILLS_DIR", str(skills_dir))

        # Clear cache
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

        # Should auto-discover from env var
        result = agent.tool.skills(action="list")
        text = extract_text(result)

        assert "code-reviewer" in text
        assert "data-analyst" in text


class TestGetSkillsPrompt:
    """Tests for get_skills_prompt helper function."""

    def test_get_skills_prompt_returns_xml(self, skills_dir):
        """Test that get_skills_prompt returns XML-formatted prompt."""
        from strands_tools.skills import _CACHE_LOCK, _cache, get_skills_prompt

        # Clear cache first
        with _CACHE_LOCK:
            _cache.clear()

        prompt = get_skills_prompt(str(skills_dir))

        assert "<available_skills>" in prompt
        assert "</available_skills>" in prompt
        assert "<skill>" in prompt
        assert "<name>code-reviewer</name>" in prompt
        assert "<name>data-analyst</name>" in prompt
        assert "<description>" in prompt

    def test_get_skills_prompt_empty_dir(self, tmp_path):
        """Test that get_skills_prompt returns empty string for empty directory."""
        from strands_tools.skills import _CACHE_LOCK, _cache, get_skills_prompt

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with _CACHE_LOCK:
            _cache.clear()

        prompt = get_skills_prompt(str(empty_dir))

        assert prompt == ""

    def test_get_skills_prompt_uses_env_var(self, skills_dir, monkeypatch):
        """Test that get_skills_prompt uses STRANDS_SKILLS_DIR env var."""
        from strands_tools.skills import _CACHE_LOCK, _cache, get_skills_prompt

        monkeypatch.setenv("STRANDS_SKILLS_DIR", str(skills_dir))

        with _CACHE_LOCK:
            _cache.clear()

        prompt = get_skills_prompt()  # No argument - should use env var

        assert "<name>code-reviewer</name>" in prompt

    def test_get_skills_prompt_shares_cache(self, agent, skills_dir):
        """Test that get_skills_prompt shares cache with skills tool."""
        from strands_tools.skills import _CACHE_LOCK, _cache, get_skills_prompt

        with _CACHE_LOCK:
            _cache.clear()

        # First call skills tool to populate cache
        agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))

        # Check cache is populated
        with _CACHE_LOCK:
            cache_size_after_tool = len(_cache)

        # Now call get_skills_prompt
        get_skills_prompt(str(skills_dir))

        # Cache size should be same (reused, not duplicated)
        with _CACHE_LOCK:
            cache_size_after_prompt = len(_cache)

        assert cache_size_after_tool == cache_size_after_prompt


class TestImportAction:
    """Tests for the import action with directory source."""

    @pytest.fixture
    def import_dir(self, tmp_path):
        """Create a second skills directory with skills to import."""
        import_path = tmp_path / "import-skills"
        import_path.mkdir()

        # Create a new skill in the import directory
        test_writer = import_path / "test-writer"
        test_writer.mkdir()
        (test_writer / "SKILL.md").write_text("""---
name: test-writer
description: Writes and manages test cases for code quality.
---

# Test Writer Skill

## Overview

This skill helps write comprehensive test cases.
""")

        return import_path

    def test_import_skills(self, agent, skills_dir, import_dir):
        """Test importing skills from another directory."""
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

        result = agent.tool.skills(
            action="import",
            source=str(import_dir),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Imported 1 skill(s)" in text
        assert "test-writer" in text

    def test_import_with_conflicts(self, agent, skills_dir, import_dir):
        """Test importing skills that already exist (conflicts)."""
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

        # Create a conflicting skill in import_dir
        conflict = import_dir / "code-reviewer"
        conflict.mkdir()
        (conflict / "SKILL.md").write_text("""---
name: code-reviewer
description: A duplicate code reviewer skill.
---

# Duplicate
""")

        # First populate cache by listing
        agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="import",
            source=str(import_dir),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Skipped 1" in text
        assert "code-reviewer" in text

    def test_import_nonexistent_directory(self, agent, skills_dir, tmp_path):
        """Test importing from a non-existent directory."""
        nonexistent = str(tmp_path / "nonexistent")

        result = agent.tool.skills(
            action="import",
            source=nonexistent,
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower()

    def test_import_missing_source(self, agent, skills_dir):
        """Test importing with no source parameter."""
        result = agent.tool.skills(
            action="import",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "source" in text.lower()


class TestImportFromFile:
    """Tests for importing a single skill from a SKILL.md file."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the skills cache before each test."""
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

    def _make_skill_file(self, tmp_path, name="my-skill", description="A test skill.", body="# Instructions\n\nDo stuff."):
        """Helper to create a standalone SKILL.md file."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(f"""---
name: {name}
description: {description}
---

{body}
""")
        return skill_file

    def test_import_from_file_success(self, agent, skills_dir, tmp_path):
        """Test importing a single skill from a SKILL.md file."""
        skill_file = self._make_skill_file(tmp_path)

        result = agent.tool.skills(
            action="import",
            source=str(skill_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "my-skill" in text

    def test_import_from_file_then_use(self, agent, skills_dir, tmp_path):
        """Test that a file-imported skill can be used."""
        skill_file = self._make_skill_file(tmp_path, body="# Review\n\nCheck for bugs.")

        agent.tool.skills(action="import", source=str(skill_file), STRANDS_SKILLS_DIR=str(skills_dir))
        result = agent.tool.skills(action="use", skill_name="my-skill", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Check for bugs" in text

    def test_import_from_file_appears_in_list(self, agent, skills_dir, tmp_path):
        """Test that a file-imported skill shows up in list."""
        skill_file = self._make_skill_file(tmp_path)

        agent.tool.skills(action="import", source=str(skill_file), STRANDS_SKILLS_DIR=str(skills_dir))
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "my-skill" in text

    def test_import_from_file_resources_accessible(self, agent, skills_dir, tmp_path):
        """Test that resources in the same directory as the SKILL.md are accessible."""
        skill_file = self._make_skill_file(tmp_path)

        # Create a resource alongside the SKILL.md
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("def helper(): pass\n")

        agent.tool.skills(action="import", source=str(skill_file), STRANDS_SKILLS_DIR=str(skills_dir))

        # list_resources should find the script
        result = agent.tool.skills(
            action="list_resources", skill_name="my-skill", STRANDS_SKILLS_DIR=str(skills_dir)
        )
        text = extract_text(result)
        assert "helper.py" in text

        # get_resource should load it
        result = agent.tool.skills(
            action="get_resource", skill_name="my-skill",
            resource_path="scripts/helper.py", STRANDS_SKILLS_DIR=str(skills_dir)
        )
        text = extract_text(result)
        assert result["status"] == "success"
        assert "def helper" in text

    def test_import_from_file_not_found(self, agent, skills_dir, tmp_path):
        """Test importing from a non-existent file."""
        result = agent.tool.skills(
            action="import",
            source=str(tmp_path / "nonexistent" / "SKILL.md"),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "not found" in extract_text(result).lower()

    def test_import_from_file_invalid_frontmatter(self, agent, skills_dir, tmp_path):
        """Test importing a file without valid frontmatter."""
        bad_file = tmp_path / "BAD.md"
        bad_file.write_text("# No frontmatter here\n\nJust markdown.")

        result = agent.tool.skills(
            action="import",
            source=str(bad_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "invalid" in extract_text(result).lower() or "frontmatter" in extract_text(result).lower()

    def test_import_from_file_missing_name(self, agent, skills_dir, tmp_path):
        """Test importing a SKILL.md with no name in frontmatter."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("""---
description: No name field here.
---

# Instructions
""")

        result = agent.tool.skills(
            action="import",
            source=str(skill_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "name" in extract_text(result).lower()

    def test_import_from_file_invalid_skill_name(self, agent, skills_dir, tmp_path):
        """Test importing a SKILL.md with an invalid (non-kebab-case) name."""
        skill_file = self._make_skill_file(tmp_path, name="Invalid_Name")

        result = agent.tool.skills(
            action="import",
            source=str(skill_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "invalid skill name" in extract_text(result).lower()

    def test_import_from_file_name_collision(self, agent, skills_dir, tmp_path):
        """Test importing a file whose skill name already exists in cache."""
        # code-reviewer already exists from skills_dir fixture
        skill_file = self._make_skill_file(tmp_path, name="code-reviewer")

        result = agent.tool.skills(
            action="import",
            source=str(skill_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "already exists" in extract_text(result).lower()

    def test_import_from_file_too_large(self, agent, skills_dir, tmp_path):
        """Test importing a file that exceeds the size limit."""
        from strands_tools.skills import MAX_SKILL_FILE_SIZE

        big_file = tmp_path / "SKILL.md"
        big_file.write_text("---\nname: big\n---\n" + "x" * (MAX_SKILL_FILE_SIZE + 1))

        result = agent.tool.skills(
            action="import",
            source=str(big_file),
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "too large" in extract_text(result).lower()


class TestImportFromUrl:
    """Tests for importing a single skill from an HTTPS URL."""

    VALID_SKILL_CONTENT = b"""---
name: remote-skill
description: A skill fetched from a URL.
---

# Remote Skill

Do remote things.
"""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the skills cache before each test."""
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

    def _mock_response(self, data=None, url="https://example.com/SKILL.md"):
        """Create a mock urllib response."""
        if data is None:
            data = self.VALID_SKILL_CONTENT
        resp = MagicMock()
        resp.read.return_value = data
        resp.url = url
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_success(self, mock_urlopen, agent, skills_dir):
        """Test importing a skill from a valid HTTPS URL."""
        mock_urlopen.return_value = self._mock_response()

        result = agent.tool.skills(
            action="import",
            source="https://example.com/skills/remote-skill/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "remote-skill" in text

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_then_use(self, mock_urlopen, agent, skills_dir):
        """Test that a URL-imported skill can be used."""
        mock_urlopen.return_value = self._mock_response()

        agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        result = agent.tool.skills(
            action="use",
            skill_name="remote-skill",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Do remote things" in text

    def test_import_from_url_http_rejected(self, agent, skills_dir):
        """Test that non-HTTPS URLs are rejected."""
        result = agent.tool.skills(
            action="import",
            source="http://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "https" in extract_text(result).lower()

    def test_import_from_url_no_hostname(self, agent, skills_dir):
        """Test that URLs without a hostname are rejected."""
        result = agent.tool.skills(
            action="import",
            source="https://",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "hostname" in extract_text(result).lower() or "invalid" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_too_large(self, mock_urlopen, agent, skills_dir):
        """Test that oversized URL content is rejected."""
        from strands_tools.skills import MAX_SKILL_FILE_SIZE

        oversized = b"x" * (MAX_SKILL_FILE_SIZE + 1)
        mock_urlopen.return_value = self._mock_response(data=oversized)

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "too large" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_http_error(self, mock_urlopen, agent, skills_dir):
        """Test handling of HTTP errors (e.g., 404)."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com/SKILL.md",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "404" in extract_text(result)

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_timeout(self, mock_urlopen, agent, skills_dir):
        """Test handling of request timeout."""
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "timeout" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_not_utf8(self, mock_urlopen, agent, skills_dir):
        """Test handling of non-UTF-8 content from URL."""
        mock_urlopen.return_value = self._mock_response(data=b"\xff\xfe\x00\x01")

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "utf-8" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_redirect_to_http(self, mock_urlopen, agent, skills_dir):
        """Test that redirect to non-HTTPS is rejected."""
        mock_urlopen.return_value = self._mock_response(url="http://evil.com/SKILL.md")

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "non-https" in extract_text(result).lower() or "redirect" in extract_text(result).lower()


class TestVirtualSkillBehavior:
    """Tests for virtual skill (URL-imported) behavior with other actions."""

    VIRTUAL_SKILL_CONTENT = b"""---
name: virtual-test
description: A virtual skill for testing.
---

# Virtual Instructions

Follow these steps.
"""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the skills cache before each test."""
        from strands_tools.skills import _CACHE_LOCK, _cache

        with _CACHE_LOCK:
            _cache.clear()

    def _mock_response(self):
        """Create a mock urllib response with the virtual skill content."""
        resp = MagicMock()
        resp.read.return_value = self.VIRTUAL_SKILL_CONTENT
        resp.url = "https://example.com/SKILL.md"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_virtual_skill_get_resource_error(self, mock_urlopen, agent, skills_dir):
        """Test that get_resource on a URL-imported skill returns a clear error."""
        mock_urlopen.return_value = self._mock_response()
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="virtual-test",
            resource_path="scripts/test.py",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "url" in text.lower() and "no local resources" in text.lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_virtual_skill_list_resources_empty(self, mock_urlopen, agent, skills_dir):
        """Test that list_resources on a URL-imported skill returns an informative message."""
        mock_urlopen.return_value = self._mock_response()
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="list_resources",
            skill_name="virtual-test",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "no local resources" in text.lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_virtual_skill_use_returns_instructions(self, mock_urlopen, agent, skills_dir):
        """Test that use action works correctly for URL-imported skills."""
        mock_urlopen.return_value = self._mock_response()
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="use",
            skill_name="virtual-test",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Follow these steps" in text


class TestExperimentalWarning:
    """Tests for the experimental warning."""

    def test_warning_emitted(self, agent, skills_dir):
        """Test that experimental warning is emitted when using the skills tool."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))

            skill_warnings = [x for x in w if "experimental" in str(x.message).lower()]
            assert len(skill_warnings) >= 1, "Expected at least one experimental warning"

    def test_warning_message_content(self, agent, skills_dir):
        """Test that the experimental warning has the expected content."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))

            skill_warnings = [x for x in w if "experimental" in str(x.message).lower()]
            assert len(skill_warnings) >= 1
            assert "native skills feature" in str(skill_warnings[0].message).lower()
