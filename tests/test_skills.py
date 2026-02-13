"""
Tests for the skills tool.

Tests cover:
- Skill listing
- Loading skill instructions
- Resource loading
- Error handling
- get_skills_prompt helper
- Import action
- Experimental warning
"""

import warnings

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
    """Tests for the import action."""

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
            import_dir=str(import_dir),
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
            import_dir=str(import_dir),
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
            import_dir=nonexistent,
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower() or "not a directory" in text.lower()

    def test_import_empty_import_dir(self, agent, skills_dir):
        """Test importing with empty import_dir parameter."""
        result = agent.tool.skills(
            action="import",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "import_dir" in text.lower()


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
