"""
Tests for the skills tool.

Tests cover:
- Skill discovery
- Skill activation/deactivation
- Resource loading
- Error handling
- Progressive disclosure pattern
"""

import os
import tempfile
from pathlib import Path

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
def agent(skills_dir):
    """Create an agent with the skills tool loaded."""
    return Agent(tools=[skills])


def extract_text(result):
    """Extract text from tool result."""
    if isinstance(result, dict):
        if "content" in result and isinstance(result["content"], list):
            return result["content"][0].get("text", "")
    return str(result)


class TestSkillDiscovery:
    """Tests for skill discovery functionality."""

    def test_discover_skills(self, agent, skills_dir):
        """Test discovering skills in a directory."""
        result = agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "Discovered 2 skill(s)" in text
        assert "code-reviewer" in text
        assert "data-analyst" in text
        # Invalid skill should not be included
        assert "Invalid_Skill" not in text

    def test_discover_empty_directory(self, agent, tmp_path):
        """Test discovering skills in an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(empty_dir))
        text = extract_text(result)

        assert "Discovered 0 skill(s)" in text

    def test_discover_nonexistent_directory(self, agent, tmp_path):
        """Test discovering skills in a nonexistent directory."""
        result = agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(tmp_path / "nonexistent"))
        text = extract_text(result)

        assert "Discovered 0 skill(s)" in text

    def test_list_skills(self, agent, skills_dir):
        """Test listing discovered skills."""
        # First discover
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        # Then list
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "code-reviewer" in text
        assert "data-analyst" in text
        assert "security" in text.lower() or "code review" in text.lower()


class TestSkillActivation:
    """Tests for skill activation/deactivation."""

    def test_activate_skill(self, agent, skills_dir):
        """Test activating a skill."""
        # Discover first
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        # Activate
        result = agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "Skill Activated: code-reviewer" in text
        assert "Guidelines" in text
        assert "security" in text.lower()

    def test_activate_nonexistent_skill(self, agent, skills_dir):
        """Test activating a skill that doesn't exist."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="activate", skill_name="nonexistent", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower()

    def test_activate_already_active_skill(self, agent, skills_dir):
        """Test activating a skill that's already active."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        # Activate again
        result = agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "already activated" in text.lower()

    def test_deactivate_skill(self, agent, skills_dir):
        """Test deactivating a skill."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="deactivate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "success"
        text = extract_text(result)
        assert "deactivated" in text.lower()

    def test_deactivate_inactive_skill(self, agent, skills_dir):
        """Test deactivating a skill that isn't active."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="deactivate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not currently activated" in text.lower()


class TestSkillResources:
    """Tests for skill resource handling."""

    def test_list_resources(self, agent, skills_dir):
        """Test listing skill resources."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="list_resources", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "scripts/" in text
        assert "analyze.py" in text
        assert "references/" in text
        assert "security.md" in text

    def test_get_resource(self, agent, skills_dir):
        """Test loading a resource file."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="scripts/analyze.py",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )
        text = extract_text(result)

        assert "analyze" in text
        assert "def analyze" in text

    def test_get_nonexistent_resource(self, agent, skills_dir):
        """Test loading a resource that doesn't exist."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="nonexistent.py",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower() or "failed" in text.lower()

    def test_get_resource_path_traversal(self, agent, skills_dir):
        """Test that path traversal is blocked."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="../data-analyst/SKILL.md",
            STRANDS_SKILLS_DIR=str(skills_dir),
        )

        assert result["status"] == "error"


class TestSkillStatus:
    """Tests for skill status functionality."""

    def test_status(self, agent, skills_dir):
        """Test getting skills status."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="status", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        assert "Discovered: 2" in text
        assert "Activated: 1" in text
        assert "code-reviewer" in text


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_action(self, agent, skills_dir):
        """Test handling of invalid action."""
        result = agent.tool.skills(action="invalid_action", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        # Validation error from Pydantic Literal type
        assert "error" in text.lower() or "validation" in text.lower()

    def test_missing_skill_name(self, agent, skills_dir):
        """Test error when skill_name is required but not provided."""
        result = agent.tool.skills(action="activate", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "skill_name is required" in text

    def test_missing_resource_path(self, agent, skills_dir):
        """Test error when resource_path is required but not provided."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))

        result = agent.tool.skills(action="get_resource", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "resource_path is required" in text


class TestProgressiveDisclosure:
    """Tests for progressive disclosure pattern."""

    def test_discovery_only_loads_metadata(self, agent, skills_dir):
        """Test that discovery only loads metadata, not full instructions."""
        result = agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        # Should have description (metadata)
        assert "security" in text.lower() or "code review" in text.lower()

        # Should NOT have full instructions content
        assert "Check for security vulnerabilities" not in text

    def test_activation_loads_full_instructions(self, agent, skills_dir):
        """Test that activation loads full instructions."""
        agent.tool.skills(action="discover", STRANDS_SKILLS_DIR=str(skills_dir))
        result = agent.tool.skills(action="activate", skill_name="code-reviewer", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        # Should have full instructions
        assert "Guidelines" in text
        assert "security" in text.lower()


class TestAutoDiscovery:
    """Tests for auto-discovery feature."""

    def test_auto_discover_on_list(self, agent, skills_dir, monkeypatch):
        """Test that skills are auto-discovered when calling list."""
        # Set the env var for auto-discover
        monkeypatch.setenv("STRANDS_SKILLS_AUTO_DISCOVER", "true")
        
        # Clear any existing registries to force new discovery
        from strands_tools.skills import _registries, _REGISTRY_LOCK
        with _REGISTRY_LOCK:
            _registries.clear()
        
        # Call list without explicit discover - should auto-discover
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        # Should have discovered skills automatically
        assert "code-reviewer" in text
        assert "data-analyst" in text

    def test_auto_discover_disabled(self, agent, skills_dir, monkeypatch):
        """Test that auto-discovery can be disabled."""
        monkeypatch.setenv("STRANDS_SKILLS_AUTO_DISCOVER", "false")
        
        # Clear any existing registries
        from strands_tools.skills import _registries, _REGISTRY_LOCK
        with _REGISTRY_LOCK:
            _registries.clear()
        
        # Call list - should NOT auto-discover
        result = agent.tool.skills(action="list", STRANDS_SKILLS_DIR=str(skills_dir))
        text = extract_text(result)

        # Should show no skills discovered
        assert "No skills discovered" in text or "Found 0 skill" in text
