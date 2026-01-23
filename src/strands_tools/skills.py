"""Agent Skills Tool for Strands Agents.

This tool provides a high-level interface for working with Agent Skills -
modular packages of instructions, scripts, and resources that give AI agents
specialized capabilities for specific tasks.

Skills follow the AgentSkills.io specification and use progressive disclosure:
- Phase 1: Load only metadata (~100 tokens per skill)
- Phase 2: Load full instructions when skill is activated
- Phase 3: Load resources (scripts, references, assets) as needed

Key features:
- Discover skills from a directory
- Activate/deactivate skills dynamically
- Load skill resources on demand
- Progressive disclosure for token efficiency

For more information about Agent Skills:
- Specification: https://agentskills.io
- Anthropic Skills: https://github.com/anthropics/skills
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Default skills directory - can be overridden via environment variable
DEFAULT_SKILLS_DIR = os.environ.get("STRANDS_SKILLS_DIR", "./skills")

# Maximum file size for resources (10MB)
MAX_RESOURCE_SIZE = 10 * 1024 * 1024


@dataclass
class SkillMetadata:
    """Lightweight skill metadata (Phase 1 of progressive disclosure)."""

    name: str
    description: str
    path: Path
    license: Optional[str] = None
    allowed_tools: Optional[List[str]] = None


@dataclass
class LoadedSkill:
    """Fully loaded skill with instructions (Phase 2)."""

    metadata: SkillMetadata
    instructions: str
    activated_at: float = 0.0


@dataclass
class SkillRegistry:
    """Registry for managing discovered and activated skills."""

    skills_dir: Path
    discovered: Dict[str, SkillMetadata] = field(default_factory=dict)
    activated: Dict[str, LoadedSkill] = field(default_factory=dict)

    def __post_init__(self):
        self.skills_dir = Path(self.skills_dir).expanduser().resolve()


# Thread-safe registry storage
_registries: Dict[str, SkillRegistry] = {}
_REGISTRY_LOCK = Lock()


def _get_or_create_registry(skills_dir: str) -> SkillRegistry:
    """Get or create a registry for the given skills directory."""
    skills_dir = str(Path(skills_dir).expanduser().resolve())
    with _REGISTRY_LOCK:
        if skills_dir not in _registries:
            _registries[skills_dir] = SkillRegistry(skills_dir=Path(skills_dir))
        return _registries[skills_dir]


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Validate that path is safely contained within base_dir.

    Prevents directory traversal attacks through symlinks or path manipulation.
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError("Invalid SKILL.md format - missing YAML frontmatter")

    frontmatter_text = match.group(1)
    body = match.group(2).strip()

    # Parse YAML frontmatter (simple key: value parsing)
    frontmatter = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Handle lists (simple format: key: item1, item2)
            if "," in value:
                value = [v.strip() for v in value.split(",")]
            frontmatter[key] = value

    return frontmatter, body


def _validate_skill_name(name: str) -> bool:
    """Validate skill name follows AgentSkills.io spec.

    - kebab-case (lowercase letters, digits, hyphens)
    - Max 64 characters
    """
    if not name or len(name) > 64:
        return False
    return bool(re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", name))


def _discover_skills(skills_dir: Path) -> Dict[str, SkillMetadata]:
    """Discover all skills in a directory (Phase 1 - metadata only)."""
    discovered = {}

    if not skills_dir.exists():
        logger.info(f"Skills directory does not exist: {skills_dir}")
        return discovered

    if not skills_dir.is_dir():
        logger.warning(f"Skills path is not a directory: {skills_dir}")
        return discovered

    for skill_folder in skills_dir.iterdir():
        if not skill_folder.is_dir():
            continue

        # Security: validate path
        if not _is_safe_path(skill_folder, skills_dir):
            logger.warning(f"Skipping unsafe path: {skill_folder}")
            continue

        # Look for SKILL.md (case-insensitive)
        skill_md = None
        for f in skill_folder.iterdir():
            if f.name.upper() == "SKILL.MD":
                skill_md = f
                break

        if not skill_md:
            logger.debug(f"No SKILL.md found in {skill_folder}")
            continue

        # Security: validate SKILL.md path
        if not _is_safe_path(skill_md, skills_dir):
            logger.warning(f"Skipping unsafe SKILL.md: {skill_md}")
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _ = _parse_frontmatter(content)

            name = frontmatter.get("name", skill_folder.name)
            description = frontmatter.get("description", "")

            if not _validate_skill_name(name):
                logger.warning(f"Invalid skill name '{name}' in {skill_folder}")
                continue

            if not description:
                logger.warning(f"Skill '{name}' has no description")

            metadata = SkillMetadata(
                name=name,
                description=description,
                path=skill_folder,
                license=frontmatter.get("license"),
                allowed_tools=frontmatter.get("allowed-tools"),
            )
            discovered[name] = metadata
            logger.debug(f"Discovered skill: {name}")

        except Exception as e:
            logger.warning(f"Error parsing skill in {skill_folder}: {e}")
            continue

    logger.info(f"Discovered {len(discovered)} skills in {skills_dir}")
    return discovered


def _load_skill_instructions(skill_path: Path) -> str:
    """Load full instructions from a skill (Phase 2)."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        # Try case-insensitive
        for f in skill_path.iterdir():
            if f.name.upper() == "SKILL.MD":
                skill_md = f
                break

    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

    content = skill_md.read_text(encoding="utf-8")
    _, instructions = _parse_frontmatter(content)
    return instructions


def _load_resource(skill_path: Path, resource_path: str) -> str:
    """Load a resource file from a skill (Phase 3).

    Args:
        skill_path: Path to the skill directory
        resource_path: Relative path to the resource (e.g., "scripts/helper.py")

    Returns:
        Content of the resource file
    """
    full_path = skill_path / resource_path

    # Security: validate path
    if not _is_safe_path(full_path, skill_path):
        raise ValueError(f"Invalid resource path: {resource_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    if not full_path.is_file():
        raise ValueError(f"Resource is not a file: {resource_path}")

    # Check file size
    if full_path.stat().st_size > MAX_RESOURCE_SIZE:
        raise ValueError(f"Resource too large (max {MAX_RESOURCE_SIZE} bytes): {resource_path}")

    return full_path.read_text(encoding="utf-8")


def _format_skills_list(skills: Dict[str, SkillMetadata], activated: Dict[str, LoadedSkill]) -> str:
    """Format skills list for display."""
    if not skills:
        return "No skills discovered."

    lines = [f"Found {len(skills)} skill(s):\n"]
    for name, metadata in sorted(skills.items()):
        status = "✓ ACTIVE" if name in activated else "○ available"
        lines.append(f"  [{status}] {name}")
        lines.append(f"      {metadata.description[:100]}{'...' if len(metadata.description) > 100 else ''}")
        lines.append(f"      Path: {metadata.path}")
        if metadata.allowed_tools:
            lines.append(f"      Allowed tools: {', '.join(metadata.allowed_tools)}")
        lines.append("")

    return "\n".join(lines)


def _format_skill_instructions(skill: LoadedSkill) -> str:
    """Format activated skill instructions for context injection."""
    return f"""
## Skill Activated: {skill.metadata.name}

{skill.metadata.description}

### Instructions

{skill.instructions}

---
Apply these guidelines when working on tasks related to "{skill.metadata.name}".
"""


# Action handlers
def _action_discover(skills_dir: str, **kwargs) -> Dict[str, Any]:
    """Discover skills in the specified directory."""
    registry = _get_or_create_registry(skills_dir)
    registry.discovered = _discover_skills(registry.skills_dir)

    return {
        "status": "success",
        "content": [
            {
                "text": f"Discovered {len(registry.discovered)} skill(s) in {skills_dir}\n\n"
                + _format_skills_list(registry.discovered, registry.activated)
            }
        ],
    }


def _action_list(skills_dir: str, **kwargs) -> Dict[str, Any]:
    """List all discovered and activated skills."""
    registry = _get_or_create_registry(skills_dir)

    # Auto-discover if not done yet
    if not registry.discovered:
        registry.discovered = _discover_skills(registry.skills_dir)

    return {
        "status": "success",
        "content": [{"text": _format_skills_list(registry.discovered, registry.activated)}],
    }


def _action_activate(skills_dir: str, skill_name: str, **kwargs) -> Dict[str, Any]:
    """Activate a skill by loading its full instructions."""
    import time

    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required for activate action"}]}

    registry = _get_or_create_registry(skills_dir)

    # Auto-discover if not done yet
    if not registry.discovered:
        registry.discovered = _discover_skills(registry.skills_dir)

    # Check if skill exists
    if skill_name not in registry.discovered:
        available = ", ".join(sorted(registry.discovered.keys())) if registry.discovered else "none"
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill_name}' not found. Available skills: {available}"}],
        }

    # Check if already activated
    if skill_name in registry.activated:
        return {
            "status": "success",
            "content": [{"text": f"Skill '{skill_name}' is already activated.\n\n" + _format_skill_instructions(registry.activated[skill_name])}],
        }

    # Load full instructions (Phase 2)
    try:
        metadata = registry.discovered[skill_name]
        instructions = _load_skill_instructions(metadata.path)

        loaded_skill = LoadedSkill(
            metadata=metadata,
            instructions=instructions,
            activated_at=time.time(),
        )
        registry.activated[skill_name] = loaded_skill

        logger.info(f"Activated skill: {skill_name}")
        return {
            "status": "success",
            "content": [{"text": _format_skill_instructions(loaded_skill)}],
        }

    except Exception as e:
        logger.error(f"Error activating skill '{skill_name}': {e}")
        return {"status": "error", "content": [{"text": f"Failed to activate skill '{skill_name}': {str(e)}"}]}


def _action_deactivate(skills_dir: str, skill_name: str, **kwargs) -> Dict[str, Any]:
    """Deactivate a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required for deactivate action"}]}

    registry = _get_or_create_registry(skills_dir)

    if skill_name not in registry.activated:
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill_name}' is not currently activated"}],
        }

    del registry.activated[skill_name]
    logger.info(f"Deactivated skill: {skill_name}")

    return {
        "status": "success",
        "content": [{"text": f"Skill '{skill_name}' has been deactivated."}],
    }


def _action_get_resource(skills_dir: str, skill_name: str, resource_path: str, **kwargs) -> Dict[str, Any]:
    """Load a resource file from a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required for get_resource action"}]}
    if not resource_path:
        return {"status": "error", "content": [{"text": "resource_path is required for get_resource action"}]}

    registry = _get_or_create_registry(skills_dir)

    # Auto-discover if not done yet
    if not registry.discovered:
        registry.discovered = _discover_skills(registry.skills_dir)

    if skill_name not in registry.discovered:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    try:
        metadata = registry.discovered[skill_name]
        content = _load_resource(metadata.path, resource_path)

        return {
            "status": "success",
            "content": [{"text": f"# Resource: {skill_name}/{resource_path}\n\n```\n{content}\n```"}],
        }

    except Exception as e:
        logger.error(f"Error loading resource '{resource_path}' from skill '{skill_name}': {e}")
        return {"status": "error", "content": [{"text": f"Failed to load resource: {str(e)}"}]}


def _action_list_resources(skills_dir: str, skill_name: str, **kwargs) -> Dict[str, Any]:
    """List available resources in a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required for list_resources action"}]}

    registry = _get_or_create_registry(skills_dir)

    # Auto-discover if not done yet
    if not registry.discovered:
        registry.discovered = _discover_skills(registry.skills_dir)

    if skill_name not in registry.discovered:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    metadata = registry.discovered[skill_name]
    skill_path = metadata.path

    resources = {"scripts": [], "references": [], "assets": [], "other": []}

    for item in skill_path.rglob("*"):
        if item.is_file() and item.name.upper() != "SKILL.MD":
            rel_path = str(item.relative_to(skill_path))

            # Categorize by directory
            if rel_path.startswith("scripts/"):
                resources["scripts"].append(rel_path)
            elif rel_path.startswith("references/"):
                resources["references"].append(rel_path)
            elif rel_path.startswith("assets/"):
                resources["assets"].append(rel_path)
            else:
                resources["other"].append(rel_path)

    lines = [f"Resources in skill '{skill_name}':\n"]

    for category, files in resources.items():
        if files:
            lines.append(f"  {category}/")
            for f in sorted(files):
                lines.append(f"    - {f}")

    if not any(resources.values()):
        lines.append("  No resources found.")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_status(skills_dir: str, **kwargs) -> Dict[str, Any]:
    """Get current skills status."""
    registry = _get_or_create_registry(skills_dir)

    lines = [
        f"Skills Directory: {registry.skills_dir}",
        f"Discovered: {len(registry.discovered)} skill(s)",
        f"Activated: {len(registry.activated)} skill(s)",
    ]

    if registry.activated:
        lines.append("\nActive Skills:")
        for name in sorted(registry.activated.keys()):
            lines.append(f"  - {name}")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


# Action dispatcher
_ACTIONS = {
    "discover": _action_discover,
    "list": _action_list,
    "activate": _action_activate,
    "deactivate": _action_deactivate,
    "get_resource": _action_get_resource,
    "list_resources": _action_list_resources,
    "status": _action_status,
}


@tool
def skills(
    action: Literal["discover", "list", "activate", "deactivate", "get_resource", "list_resources", "status"],
    skill_name: Optional[str] = None,
    resource_path: Optional[str] = None,
    skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Manage Agent Skills - modular packages of specialized knowledge and instructions.

    Agent Skills are folders containing instructions, scripts, and resources that teach
    you how to perform specific tasks effectively. Skills follow the AgentSkills.io
    specification and use progressive disclosure for token efficiency.

    Supported actions:
    - discover: Scan a directory to find available skills
    - list: Show all discovered skills and their activation status
    - activate: Load a skill's full instructions into your context
    - deactivate: Remove a skill from your active context
    - get_resource: Load a specific resource file from a skill (scripts, references, etc.)
    - list_resources: List all resource files available in a skill
    - status: Show current skills directory and activation status

    Args:
        action: The action to perform
        skill_name: Name of the skill (required for activate, deactivate, get_resource, list_resources)
        resource_path: Path to resource file relative to skill directory (required for get_resource)
        skills_dir: Directory containing skills (default: ./skills or STRANDS_SKILLS_DIR env var)

    Returns:
        Dict with status and content describing the result

    Examples:
        # Discover skills in the default directory
        skills(action="discover")

        # List all available skills
        skills(action="list")

        # Activate a skill to load its instructions
        skills(action="activate", skill_name="code-reviewer")

        # List resources in a skill
        skills(action="list_resources", skill_name="pdf-processor")

        # Load a specific resource
        skills(action="get_resource", skill_name="pdf-processor", resource_path="scripts/extract.py")

        # Deactivate a skill when done
        skills(action="deactivate", skill_name="code-reviewer")

        # Check current status
        skills(action="status")
    """
    # Resolve skills directory
    skills_dir = skills_dir or os.environ.get("STRANDS_SKILLS_DIR", DEFAULT_SKILLS_DIR)

    # Validate action
    if action not in _ACTIONS:
        valid_actions = ", ".join(_ACTIONS.keys())
        return {"status": "error", "content": [{"text": f"Invalid action '{action}'. Valid actions: {valid_actions}"}]}

    try:
        # Dispatch to action handler
        handler = _ACTIONS[action]
        return handler(
            skills_dir=skills_dir,
            skill_name=skill_name,
            resource_path=resource_path,
        )

    except Exception as e:
        logger.error(f"Error in skills tool action '{action}': {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
