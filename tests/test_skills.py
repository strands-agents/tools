"""Tests for the skills tool.

Tests cover:
- Skill listing
- Loading skill instructions
- Resource loading
- Error handling
- Import action (directory, file, URL)
- Virtual skill behavior
- SDK plugin info in docstring
- Agent state caching
- Comma-separated skills directories
- Tool spec update with skills catalog
- Skill resolution (individual dir, parent dir, SKILL.md file)
- Format skill response (metadata, resources)
"""

import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.vended_plugins.skills.skill import Skill

from strands_tools import skills
from strands_tools.skills import (
    MAX_RESOURCE_SIZE,
    MAX_SKILL_FILE_SIZE,
    _cache,
    _format_skill_response,
    _generate_skills_xml,
    _list_skill_resources,
    _resolve_skills,
    sync_skills,
)
from strands_tools.skills import (
    skills as skills_tool,
)


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


def clear_cache():
    """Clear the module-level skills cache."""
    _cache.clear()


@pytest.fixture(autouse=True)
def _clear_cache_before_each_test():
    """Clear the module-level skills cache before every test."""
    clear_cache()


class TestSkillListing:
    """Tests for skill listing functionality."""

    def test_list_skills(self, agent, skills_dir):
        """Test listing skills in a directory."""
        result = agent.tool.skills(action="list", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert "code-reviewer" in text
        assert "data-analyst" in text
        assert "Available skills" in text

    def test_list_empty_directory(self, agent, tmp_path):
        """Test listing skills in an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = agent.tool.skills(action="list", skills_dir=str(empty_dir))
        text = extract_text(result)

        assert "No skills found" in text

    def test_list_nonexistent_directory(self, agent, tmp_path):
        """Test listing skills in a nonexistent directory."""
        nonexistent = str(tmp_path / "nonexistent")
        result = agent.tool.skills(action="list", skills_dir=nonexistent)
        text = extract_text(result)

        assert "No skills found" in text


class TestSkillUsage:
    """Tests for loading skill instructions."""

    def test_use_skill(self, agent, skills_dir):
        """Test loading a skill's instructions."""
        result = agent.tool.skills(action="use", skill_name="code-reviewer", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Code Reviewer Skill" in text
        assert "security vulnerabilities" in text.lower()

    def test_use_skill_includes_metadata(self, agent, skills_dir):
        """Test that use action includes metadata like location."""
        result = agent.tool.skills(action="use", skill_name="code-reviewer", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert "Location:" in text
        assert "SKILL.md" in text

    def test_use_skill_includes_resources(self, agent, skills_dir):
        """Test that use action lists available resources."""
        result = agent.tool.skills(action="use", skill_name="code-reviewer", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert "Available resources:" in text
        assert "scripts/analyze.py" in text
        assert "references/security.md" in text

    def test_use_nonexistent_skill(self, agent, skills_dir):
        """Test loading a skill that doesn't exist."""
        result = agent.tool.skills(action="use", skill_name="nonexistent", skills_dir=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower()

    def test_use_skill_missing_name(self, agent, skills_dir):
        """Test error when skill_name is not provided."""
        result = agent.tool.skills(action="use", skills_dir=str(skills_dir))

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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"

    def test_get_resource_path_traversal(self, agent, skills_dir):
        """Test that path traversal is blocked."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="../data-analyst/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_action(self, agent, skills_dir):
        """Test handling of invalid action."""
        result = agent.tool.skills(action="invalid_action", skills_dir=str(skills_dir))

        assert result["status"] == "error"
        text = extract_text(result)
        assert "error" in text.lower() or "invalid" in text.lower()

    def test_missing_resource_path(self, agent, skills_dir):
        """Test error when resource_path is required but not provided."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "resource_path is required" in text


class TestAutoDiscovery:
    """Tests for auto-discovery feature."""

    def test_auto_discover_from_env(self, agent, skills_dir, monkeypatch):
        """Test that skills are auto-discovered from env var."""
        monkeypatch.setenv("STRANDS_SKILLS_DIR", str(skills_dir))
        clear_cache()

        result = agent.tool.skills(action="list")
        text = extract_text(result)

        assert "code-reviewer" in text
        assert "data-analyst" in text


class TestCommaSeparatedDirs:
    """Tests for comma-separated STRANDS_SKILLS_DIR support."""

    def test_comma_separated_dirs(self, agent, tmp_path):
        """Test that comma-separated directories are all scanned."""
        clear_cache()

        dir_a = tmp_path / "dir-a"
        dir_a.mkdir()
        skill_a = dir_a / "skill-a"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: skill-a\ndescription: From dir A.\n---\n# A\n")

        dir_b = tmp_path / "dir-b"
        dir_b.mkdir()
        skill_b = dir_b / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: skill-b\ndescription: From dir B.\n---\n# B\n")

        result = agent.tool.skills(action="list", skills_dir=f"{dir_a},{dir_b}")
        text = extract_text(result)

        assert "skill-a" in text
        assert "skill-b" in text

    def test_comma_separated_with_spaces(self, agent, tmp_path):
        """Test that spaces around commas are trimmed."""
        clear_cache()

        dir_a = tmp_path / "dir-a"
        dir_a.mkdir()
        skill_a = dir_a / "skill-a"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: skill-a\ndescription: From dir A.\n---\n# A\n")

        result = agent.tool.skills(action="list", skills_dir=f"  {dir_a}  , {tmp_path / 'nonexistent'}  ")
        text = extract_text(result)

        assert "skill-a" in text

    def test_comma_separated_env_var(self, agent, tmp_path, monkeypatch):
        """Test comma-separated dirs via environment variable."""
        clear_cache()

        dir_a = tmp_path / "dir-a"
        dir_a.mkdir()
        skill_a = dir_a / "skill-a"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: skill-a\ndescription: From dir A.\n---\n# A\n")

        dir_b = tmp_path / "dir-b"
        dir_b.mkdir()
        skill_b = dir_b / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: skill-b\ndescription: From dir B.\n---\n# B\n")

        monkeypatch.setenv("STRANDS_SKILLS_DIR", f"{dir_a},{dir_b}")

        result = agent.tool.skills(action="list")
        text = extract_text(result)

        assert "skill-a" in text
        assert "skill-b" in text

    def test_comma_separated_duplicate_override(self, agent, tmp_path):
        """Test that later directories override earlier ones for duplicate names."""
        clear_cache()

        dir_a = tmp_path / "dir-a"
        dir_a.mkdir()
        skill_dir_a = dir_a / "dupe"
        skill_dir_a.mkdir()
        (skill_dir_a / "SKILL.md").write_text("---\nname: dupe\ndescription: From dir A.\n---\n# A instructions\n")

        dir_b = tmp_path / "dir-b"
        dir_b.mkdir()
        skill_dir_b = dir_b / "dupe"
        skill_dir_b.mkdir()
        (skill_dir_b / "SKILL.md").write_text("---\nname: dupe\ndescription: From dir B.\n---\n# B instructions\n")

        result = agent.tool.skills(action="list", skills_dir=f"{dir_a},{dir_b}")
        text = extract_text(result)
        assert "dupe" in text

        # The second dir should win (overwrite)
        result = agent.tool.skills(action="use", skill_name="dupe", skills_dir=f"{dir_a},{dir_b}")
        text = extract_text(result)
        assert "B instructions" in text


class TestResolveSkills:
    """Tests for _resolve_skills logic."""

    def test_resolve_parent_directory(self, tmp_path):
        """Test resolving a parent directory containing skill subdirs."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: Test.\n---\n# Hi\n")

        result = _resolve_skills([str(tmp_path)])
        assert "my-skill" in result

    def test_resolve_individual_skill_dir(self, tmp_path):
        """Test resolving a single skill directory (has SKILL.md)."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: Test.\n---\n# Hi\n")

        result = _resolve_skills([str(skill_dir)])
        assert "my-skill" in result

    def test_resolve_skill_md_file(self, tmp_path):
        """Test resolving a direct path to a SKILL.md file."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: my-skill\ndescription: Test.\n---\n# Hi\n")

        result = _resolve_skills([str(skill_file)])
        assert "my-skill" in result

    def test_resolve_nonexistent_path(self, tmp_path):
        """Test that nonexistent paths are skipped."""
        result = _resolve_skills([str(tmp_path / "nonexistent")])
        assert len(result) == 0

    def test_resolve_mixed_sources(self, tmp_path):
        """Test resolving a mix of source types."""
        # Parent dir with one skill
        parent = tmp_path / "parent"
        parent.mkdir()
        s1 = parent / "skill-one"
        s1.mkdir()
        (s1 / "SKILL.md").write_text("---\nname: skill-one\ndescription: One.\n---\n# One\n")

        # Individual skill dir
        s2 = tmp_path / "skill-two"
        s2.mkdir()
        (s2 / "SKILL.md").write_text("---\nname: skill-two\ndescription: Two.\n---\n# Two\n")

        result = _resolve_skills([str(parent), str(s2)])
        assert "skill-one" in result
        assert "skill-two" in result


class TestGenerateSkillsXml:
    """Tests for _generate_skills_xml."""

    def test_empty_cache(self):
        """Test XML generation with no skills."""
        xml = _generate_skills_xml({})
        assert "<available_skills>" in xml
        assert "No skills are currently available" in xml

    def test_with_skills(self, tmp_path):
        """Test XML generation with skills."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill = Skill(name="test-skill", description="A test skill.", path=skill_dir)

        xml = _generate_skills_xml({"test-skill": skill})
        assert "<name>test-skill</name>" in xml
        assert "<description>A test skill.</description>" in xml
        assert "<location>" in xml
        assert "SKILL.md</location>" in xml

    def test_virtual_skill_no_location(self):
        """Test XML generation for virtual skill (no path)."""
        skill = Skill(name="virtual", description="Virtual skill.", path=None)

        xml = _generate_skills_xml({"virtual": skill})
        assert "<name>virtual</name>" in xml
        assert "<location>" not in xml

    def test_xml_escaping(self):
        """Test that special characters are escaped in XML."""
        skill = Skill(name="test", description='Uses <tags> & "quotes".', path=None)

        xml = _generate_skills_xml({"test": skill})
        assert "&lt;tags&gt;" in xml
        assert "&amp;" in xml


class TestFormatSkillResponse:
    """Tests for _format_skill_response."""

    def test_basic_response(self):
        """Test basic skill response formatting."""
        skill = Skill(name="test", description="Test.", instructions="Do the thing.", path=None)
        result = _format_skill_response(skill)
        assert "Do the thing." in result

    def test_no_instructions(self):
        """Test response when skill has no instructions."""
        skill = Skill(name="test", description="Test.", instructions="", path=None)
        result = _format_skill_response(skill)
        assert "no instructions available" in result.lower()

    def test_includes_allowed_tools(self):
        """Test that allowed_tools are included in response."""
        skill = Skill(
            name="test",
            description="Test.",
            instructions="Do stuff.",
            allowed_tools=["file_read", "http_request"],
            path=None,
        )
        result = _format_skill_response(skill)
        assert "Allowed tools: file_read, http_request" in result

    def test_includes_compatibility(self):
        """Test that compatibility info is included in response."""
        skill = Skill(
            name="test",
            description="Test.",
            instructions="Do stuff.",
            compatibility="Python 3.10+",
            path=None,
        )
        result = _format_skill_response(skill)
        assert "Compatibility: Python 3.10+" in result

    def test_includes_location(self, tmp_path):
        """Test that location is included for filesystem skills."""
        skill = Skill(name="test", description="Test.", instructions="Do stuff.", path=tmp_path)
        result = _format_skill_response(skill)
        assert "Location:" in result
        assert "SKILL.md" in result

    def test_includes_resources(self, tmp_path):
        """Test that resources are listed for filesystem skills."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("pass")

        skill = Skill(name="test", description="Test.", instructions="Do stuff.", path=tmp_path)
        result = _format_skill_response(skill)
        assert "Available resources:" in result
        assert "scripts/helper.py" in result


class TestListSkillResources:
    """Tests for _list_skill_resources."""

    def test_lists_scripts_references_assets(self, tmp_path):
        """Test that all three resource dirs are scanned."""
        for d in ("scripts", "references", "assets"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "file.txt").write_text("content")

        result = _list_skill_resources(tmp_path)
        assert "scripts/file.txt" in result
        assert "references/file.txt" in result
        assert "assets/file.txt" in result

    def test_empty_skill_dir(self, tmp_path):
        """Test with no resource directories."""
        result = _list_skill_resources(tmp_path)
        assert result == []

    def test_truncation(self, tmp_path):
        """Test that results are truncated at the limit."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        for i in range(25):
            (scripts_dir / f"file_{i:02d}.py").write_text("pass")

        result = _list_skill_resources(tmp_path)
        assert "truncated" in result[-1].lower()


class TestUpdateToolSpec:
    """Tests for _update_tool_spec."""

    def test_updates_tool_description(self, agent, skills_dir):
        """Test that tool spec is updated with skills catalog after invocation."""
        clear_cache()
        agent.tool.skills(action="list", skills_dir=str(skills_dir))

        desc = skills_tool.tool_spec["description"]
        assert "<available_skills>" in desc
        assert "code-reviewer" in desc
        assert "data-analyst" in desc

    def test_updates_tool_spec_preserves_name_and_schema(self, agent, skills_dir):
        """Test that tool spec update preserves name and inputSchema."""
        original_name = skills_tool.tool_spec["name"]
        original_schema = skills_tool.tool_spec["inputSchema"]

        clear_cache()
        agent.tool.skills(action="list", skills_dir=str(skills_dir))

        assert skills_tool.tool_spec["name"] == original_name
        assert skills_tool.tool_spec["inputSchema"] == original_schema


class TestImportAction:
    """Tests for the import action with directory source."""

    @pytest.fixture
    def import_dir(self, tmp_path):
        """Create a second skills directory with skills to import."""
        import_path = tmp_path / "import-skills"
        import_path.mkdir()

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
        clear_cache()

        result = agent.tool.skills(
            action="import",
            source=str(import_dir),
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Imported 1 skill(s)" in text
        assert "test-writer" in text

    def test_import_with_conflicts(self, agent, skills_dir, import_dir):
        """Test importing skills that already exist (conflicts)."""
        clear_cache()

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
        agent.tool.skills(action="list", skills_dir=str(skills_dir))

        result = agent.tool.skills(
            action="import",
            source=str(import_dir),
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "not found" in text.lower()

    def test_import_missing_source(self, agent, skills_dir):
        """Test importing with no source parameter."""
        result = agent.tool.skills(
            action="import",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "source" in text.lower()


class TestImportFromFile:
    """Tests for importing a single skill from a SKILL.md file."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the skills cache before each test."""
        clear_cache()

    def _make_skill_file(
        self, tmp_path, name="my-skill", description="A test skill.", body="# Instructions\n\nDo stuff."
    ):
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
        # Use a completely separate temp directory so the file isn't auto-discovered from skills_dir
        with tempfile.TemporaryDirectory() as import_dir:
            skill_file = self._make_skill_file(Path(import_dir))

            result = agent.tool.skills(
                action="import",
                source=str(skill_file),
                skills_dir=str(skills_dir),
            )
            text = extract_text(result)

            assert result["status"] == "success"
            assert "my-skill" in text

    def test_import_from_file_then_use(self, agent, skills_dir, tmp_path):
        """Test that a file-imported skill can be used."""
        skill_file = self._make_skill_file(tmp_path, body="# Review\n\nCheck for bugs.")

        agent.tool.skills(action="import", source=str(skill_file), skills_dir=str(skills_dir))
        result = agent.tool.skills(action="use", skill_name="my-skill", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Check for bugs" in text

    def test_import_from_file_appears_in_list(self, agent, skills_dir, tmp_path):
        """Test that a file-imported skill shows up in list."""
        skill_file = self._make_skill_file(tmp_path)

        agent.tool.skills(action="import", source=str(skill_file), skills_dir=str(skills_dir))
        result = agent.tool.skills(action="list", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert "my-skill" in text

    def test_import_from_file_resources_accessible(self, agent, skills_dir, tmp_path):
        """Test that resources in the same directory as the SKILL.md are accessible."""
        skill_file = self._make_skill_file(tmp_path)

        # Create a resource alongside the SKILL.md
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("def helper(): pass\n")

        agent.tool.skills(action="import", source=str(skill_file), skills_dir=str(skills_dir))

        # list_resources should find the script
        result = agent.tool.skills(action="list_resources", skill_name="my-skill", skills_dir=str(skills_dir))
        text = extract_text(result)
        assert "helper.py" in text

        # get_resource should load it
        result = agent.tool.skills(
            action="get_resource",
            skill_name="my-skill",
            resource_path="scripts/helper.py",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)
        assert result["status"] == "success"
        assert "def helper" in text

    def test_import_from_file_not_found(self, agent, skills_dir, tmp_path):
        """Test importing from a non-existent file."""
        result = agent.tool.skills(
            action="import",
            source=str(tmp_path / "nonexistent" / "SKILL.md"),
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "name" in extract_text(result).lower()

    def test_import_from_file_name_collision(self, agent, skills_dir, tmp_path):
        """Test importing a file whose skill name already exists in cache."""
        # code-reviewer already exists from skills_dir fixture
        skill_file = self._make_skill_file(tmp_path, name="code-reviewer")

        result = agent.tool.skills(
            action="import",
            source=str(skill_file),
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "already exists" in extract_text(result).lower()

    def test_import_from_file_too_large(self, agent, skills_dir, tmp_path):
        """Test importing a file that exceeds the size limit."""
        big_file = tmp_path / "SKILL.md"
        big_file.write_text("---\nname: big\ndescription: big skill\n---\n" + "x" * (MAX_SKILL_FILE_SIZE + 1))

        result = agent.tool.skills(
            action="import",
            source=str(big_file),
            skills_dir=str(skills_dir),
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
        clear_cache()

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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
        )
        result = agent.tool.skills(
            action="use",
            skill_name="remote-skill",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Do remote things" in text

    def test_import_from_url_http_rejected(self, agent, skills_dir):
        """Test that non-HTTPS URLs are rejected."""
        result = agent.tool.skills(
            action="import",
            source="http://example.com/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "https" in extract_text(result).lower()

    def test_import_from_url_no_hostname(self, agent, skills_dir):
        """Test that URLs without a hostname are rejected."""
        result = agent.tool.skills(
            action="import",
            source="https://",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "hostname" in extract_text(result).lower() or "invalid" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_too_large(self, mock_urlopen, agent, skills_dir):
        """Test that oversized URL content is rejected."""
        oversized = b"x" * (MAX_SKILL_FILE_SIZE + 1)
        mock_urlopen.return_value = self._mock_response(data=oversized)

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "too large" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_http_error(self, mock_urlopen, agent, skills_dir):
        """Test handling of HTTP errors (e.g., 404)."""
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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
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
            skills_dir=str(skills_dir),
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
        clear_cache()

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
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", skills_dir=str(skills_dir))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="virtual-test",
            resource_path="scripts/test.py",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        text = extract_text(result)
        assert "url" in text.lower() and "no local resources" in text.lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_virtual_skill_list_resources_empty(self, mock_urlopen, agent, skills_dir):
        """Test that list_resources on a URL-imported skill returns an informative message."""
        mock_urlopen.return_value = self._mock_response()
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", skills_dir=str(skills_dir))

        result = agent.tool.skills(
            action="list_resources",
            skill_name="virtual-test",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "no local resources" in text.lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_virtual_skill_use_returns_instructions(self, mock_urlopen, agent, skills_dir):
        """Test that use action works correctly for URL-imported skills."""
        mock_urlopen.return_value = self._mock_response()
        agent.tool.skills(action="import", source="https://example.com/SKILL.md", skills_dir=str(skills_dir))

        result = agent.tool.skills(
            action="use",
            skill_name="virtual-test",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "Follow these steps" in text


class TestLoadResourceEdgeCases:
    """Tests for _load_resource edge cases."""

    def test_resource_is_directory(self, agent, skills_dir):
        """Test that loading a directory as a resource returns an error."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="scripts",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "not a file" in extract_text(result).lower()

    def test_resource_too_large(self, agent, skills_dir):
        """Test that loading a resource exceeding MAX_RESOURCE_SIZE returns an error."""
        large_file = skills_dir / "code-reviewer" / "scripts" / "big.txt"
        large_file.write_text("x" * (MAX_RESOURCE_SIZE + 1))

        result = agent.tool.skills(
            action="get_resource",
            skill_name="code-reviewer",
            resource_path="scripts/big.txt",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "too large" in extract_text(result).lower()


class TestImportFromFileEdgeCases:
    """Tests for _import_from_file edge cases."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        clear_cache()

    def test_import_from_file_not_utf8(self, agent, skills_dir, tmp_path):
        """Test importing a file that is not valid UTF-8."""
        bad_file = tmp_path / "SKILL.md"
        bad_file.write_bytes(b"\xff\xfe\x00\x01")

        result = agent.tool.skills(
            action="import",
            source=str(bad_file),
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "utf-8" in extract_text(result).lower()

    def test_import_from_file_is_directory(self, agent, skills_dir, tmp_path):
        """Test importing when source points to a directory but is_file check fails."""
        empty = tmp_path / "empty-dir"
        empty.mkdir()

        result = agent.tool.skills(
            action="import",
            source=str(empty),
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        # Should be handled by _import_from_directory (0 skills imported)
        assert result["status"] == "success"
        assert "Imported 0 skill(s)" in text


class TestImportSchemeRejection:
    """Tests for rejecting non-HTTPS URL schemes in import."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        clear_cache()

    def test_import_ftp_rejected(self, agent, skills_dir):
        """Test that ftp:// URLs are rejected."""
        result = agent.tool.skills(
            action="import",
            source="ftp://example.com/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "https" in extract_text(result).lower()

    def test_import_file_scheme_rejected(self, agent, skills_dir):
        """Test that file:// URLs are rejected."""
        result = agent.tool.skills(
            action="import",
            source="file:///etc/passwd",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "https" in extract_text(result).lower()


class TestImportFromUrlEdgeCases:
    """Tests for _import_from_url edge cases."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        clear_cache()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_urlerror(self, mock_urlopen, agent, skills_dir):
        """Test handling of URLError (e.g., DNS failure)."""
        mock_urlopen.side_effect = urllib.error.URLError("Name resolution failed")

        result = agent.tool.skills(
            action="import",
            source="https://nonexistent.example.com/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "error fetching url" in extract_text(result).lower()

    @patch("strands_tools.skills.urllib.request.urlopen")
    def test_import_from_url_oserror(self, mock_urlopen, agent, skills_dir):
        """Test handling of generic OSError during URL fetch."""
        mock_urlopen.side_effect = OSError("Network is unreachable")

        result = agent.tool.skills(
            action="import",
            source="https://example.com/SKILL.md",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "network error" in extract_text(result).lower()


class TestListResourcesEdgeCases:
    """Tests for _action_list_resources edge cases."""

    def test_list_resources_missing_skill_name(self, agent, skills_dir):
        """Test list_resources with no skill_name."""
        result = agent.tool.skills(
            action="list_resources",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "skill_name is required" in extract_text(result)

    def test_list_resources_nonexistent_skill(self, agent, skills_dir):
        """Test list_resources for a skill that doesn't exist."""
        result = agent.tool.skills(
            action="list_resources",
            skill_name="nonexistent",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "not found" in extract_text(result).lower()

    def test_list_resources_no_resources(self, agent, skills_dir):
        """Test list_resources for a skill with no resource files."""
        result = agent.tool.skills(
            action="list_resources",
            skill_name="data-analyst",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert result["status"] == "success"
        assert "No resources found" in text

    def test_list_resources_assets_and_other(self, agent, skills_dir):
        """Test that assets/ and uncategorized files are listed correctly."""
        skill_path = skills_dir / "code-reviewer"

        assets_dir = skill_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "logo.png").write_text("fake image")

        (skill_path / "README.txt").write_text("readme content")

        result = agent.tool.skills(
            action="list_resources",
            skill_name="code-reviewer",
            skills_dir=str(skills_dir),
        )
        text = extract_text(result)

        assert "assets/" in text
        assert "logo.png" in text
        assert "other/" in text or "README.txt" in text


class TestGetResourceEdgeCases:
    """Tests for _action_get_resource edge cases."""

    def test_get_resource_missing_skill_name(self, agent, skills_dir):
        """Test get_resource with no skill_name."""
        result = agent.tool.skills(
            action="get_resource",
            resource_path="scripts/analyze.py",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "skill_name is required" in extract_text(result)

    def test_get_resource_nonexistent_skill(self, agent, skills_dir):
        """Test get_resource for a skill that doesn't exist."""
        result = agent.tool.skills(
            action="get_resource",
            skill_name="nonexistent",
            resource_path="foo.py",
            skills_dir=str(skills_dir),
        )

        assert result["status"] == "error"
        assert "not found" in extract_text(result).lower()


class TestUseSkillEdgeCases:
    """Tests for _action_use edge cases."""

    def test_use_skill_corrupted_skill_md(self, agent, tmp_path):
        """Test use action when SKILL.md is deleted after discovery."""
        clear_cache()

        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: broken-skill
description: Will break.
---

# Instructions
""")

        # Discover the skill
        result = agent.tool.skills(action="list", skills_dir=str(tmp_path))
        assert "broken-skill" in extract_text(result)

        # Now delete the SKILL.md — the SDK already loaded instructions at discovery,
        # so use should still work (instructions are in the Skill object)
        skill_md.unlink()

        result = agent.tool.skills(action="use", skill_name="broken-skill", skills_dir=str(tmp_path))
        assert result["status"] == "success"

    def test_use_skill_available_list_on_not_found(self, agent, skills_dir):
        """Test that use action lists available skills when skill not found."""
        result = agent.tool.skills(action="use", skill_name="nonexistent", skills_dir=str(skills_dir))
        text = extract_text(result)

        assert result["status"] == "error"
        assert "code-reviewer" in text or "data-analyst" in text


class TestListDescriptionTruncation:
    """Tests for description truncation and allowed_tools display in list."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        clear_cache()

    def test_list_truncates_long_description(self, agent, tmp_path):
        """Test that descriptions longer than 100 chars are truncated."""
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        long_desc = "A" * 150
        (skill_dir / "SKILL.md").write_text(f"""---
name: long-desc
description: {long_desc}
---

# Instructions
""")

        result = agent.tool.skills(action="list", skills_dir=str(tmp_path))
        text = extract_text(result)

        assert "..." in text
        assert long_desc not in text

    def test_list_shows_allowed_tools(self, agent, tmp_path):
        """Test that allowed_tools are displayed in list output."""
        skill_dir = tmp_path / "tooled"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: tooled
description: A skill with allowed tools.
allowed-tools:
  - file_read
  - http_request
---

# Instructions
""")

        result = agent.tool.skills(action="list", skills_dir=str(tmp_path))
        text = extract_text(result)

        assert "Allowed tools:" in text
        assert "file_read" in text
        assert "http_request" in text


class TestModuleCache:
    """Tests for module-level caching."""

    def test_cache_persists_across_calls(self, agent, skills_dir):
        """Test that cache persists across tool calls."""
        clear_cache()
        agent.tool.skills(action="list", skills_dir=str(skills_dir))

        assert "code-reviewer" in _cache

        result = agent.tool.skills(action="list", skills_dir=str(skills_dir))
        assert "code-reviewer" in extract_text(result)

    def test_cache_cleared_allows_rediscovery(self, agent, skills_dir):
        """Test that clearing cache allows fresh discovery."""
        agent.tool.skills(action="list", skills_dir=str(skills_dir))
        assert "code-reviewer" in _cache

        clear_cache()
        assert len(_cache) == 0

        result = agent.tool.skills(action="list", skills_dir=str(skills_dir))
        assert "code-reviewer" in extract_text(result)

    def test_imported_skill_in_cache(self, agent, skills_dir, tmp_path):
        """Test that imported skills are stored in module cache."""
        clear_cache()

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("""---
name: state-test
description: Testing state storage.
---

# Instructions
""")

        agent.tool.skills(action="import", source=str(skill_file), skills_dir=str(skills_dir))

        assert "state-test" in _cache


class TestTopLevelExceptionHandler:
    """Tests for the top-level exception handler in skills()."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        clear_cache()

    def test_unexpected_exception_caught(self, agent, skills_dir):
        """Test that unexpected exceptions in action handlers are caught."""
        with patch.dict("strands_tools.skills._ACTIONS", {"list": MagicMock(side_effect=RuntimeError("boom"))}):
            result = agent.tool.skills(action="list", skills_dir=str(skills_dir))

            assert result["status"] == "error"
            assert "boom" in extract_text(result).lower()


class TestSyncSkills:
    """Tests for the sync_skills public helper."""

    def test_populates_cache(self, agent, skills_dir):
        """Test that sync_skills populates the module-level cache."""
        clear_cache()
        assert len(_cache) == 0

        sync_skills(skills_dir=str(skills_dir))

        assert "code-reviewer" in _cache
        assert "data-analyst" in _cache

    def test_updates_tool_spec(self, agent, skills_dir):
        """Test that sync_skills updates the skills tool description."""
        clear_cache()
        sync_skills(skills_dir=str(skills_dir))

        desc = skills_tool.tool_spec["description"]
        assert "<available_skills>" in desc
        assert "code-reviewer" in desc

    def test_uses_env_var_default(self, agent, skills_dir, monkeypatch):
        """Test that sync_skills falls back to STRANDS_SKILLS_DIR env var."""
        clear_cache()
        monkeypatch.setenv("STRANDS_SKILLS_DIR", str(skills_dir))

        sync_skills()

        assert "code-reviewer" in _cache

    def test_comma_separated_dirs(self, agent, tmp_path):
        """Test that sync_skills handles comma-separated directories."""
        clear_cache()

        dir_a = tmp_path / "dir-a"
        dir_a.mkdir()
        s = dir_a / "skill-a"
        s.mkdir()
        (s / "SKILL.md").write_text("---\nname: skill-a\ndescription: A.\n---\n# A\n")

        dir_b = tmp_path / "dir-b"
        dir_b.mkdir()
        s = dir_b / "skill-b"
        s.mkdir()
        (s / "SKILL.md").write_text("---\nname: skill-b\ndescription: B.\n---\n# B\n")

        sync_skills(skills_dir=f"{dir_a},{dir_b}")

        assert "skill-a" in _cache
        assert "skill-b" in _cache

    def test_empty_directory(self, agent, tmp_path):
        """Test sync_skills with an empty directory."""
        clear_cache()
        empty = tmp_path / "empty"
        empty.mkdir()

        sync_skills(skills_dir=str(empty))

        assert len(_cache) == 0

    def test_does_not_overwrite_existing_cache(self, agent, skills_dir, tmp_path):
        """Test that sync_skills merges into existing cache."""
        clear_cache()

        # Pre-populate cache with a skill
        pre_skill = Skill(name="pre-existing", description="Already here.", path=None)
        _cache["pre-existing"] = pre_skill

        sync_skills(skills_dir=str(skills_dir))

        assert "pre-existing" in _cache
        assert "code-reviewer" in _cache

    def test_subsequent_tool_call_uses_preloaded_cache(self, agent, skills_dir):
        """Test that tool calls after sync_skills use the pre-populated cache."""
        clear_cache()
        sync_skills(skills_dir=str(skills_dir))

        result = agent.tool.skills(action="use", skill_name="code-reviewer", skills_dir=str(skills_dir))
        assert result["status"] == "success"
        assert "Code Reviewer Skill" in extract_text(result)
