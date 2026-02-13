"""Agent Skills Tool for Strands Agents.

This tool provides a high-level interface for working with Agent Skills -
modular packages of instructions, scripts, and resources that give AI agents
specialized capabilities for specific tasks.

Skills follow the AgentSkills.io specification and use progressive disclosure:
- Phase 1: Load only metadata (~100 tokens per skill) via 'list' action
- Phase 2: Load full instructions when skill is used via 'use' action
- Phase 3: Load resources (scripts, references, assets) as needed via 'get_resource'

Environment Variables:
    STRANDS_SKILLS_DIR: Directory containing skills (default: ./skills)

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools import skills

# Set skills directory via env var or parameter
import os
os.environ["STRANDS_SKILLS_DIR"] = "./my-skills"

agent = Agent(tools=[skills])

# List available skills
agent.tool.skills(action="list")

# Use a skill - loads its instructions
agent.tool.skills(action="use", skill_name="code-reviewer")

# Get a resource file from a skill
agent.tool.skills(action="get_resource", skill_name="code-reviewer", resource_path="scripts/analyze.py")
```

For more information about Agent Skills:
- Specification: https://agentskills.io
- Anthropic Skills: https://github.com/anthropics/skills
"""

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

import yaml
from strands import tool

logger = logging.getLogger(__name__)
logger.warning(
    "The skills tool is experimental. The recommended path for production use will be "
    "the SDK's native skills feature. APIs and behavior may change without notice."
)

# Maximum file size for resources (10MB)
MAX_RESOURCE_SIZE = 10 * 1024 * 1024


@dataclass
class SkillMetadata:
    """Skill metadata from SKILL.md frontmatter."""

    name: str
    description: str
    path: Path
    license: Optional[str] = None
    allowed_tools: Optional[List[str]] = None


@dataclass
class SkillsCache:
    """Cache for discovered skills in a directory."""

    skills_dir: Path
    skills: Dict[str, SkillMetadata] = field(default_factory=dict)


# Thread-safe cache storage
_cache: Dict[str, SkillsCache] = {}
_CACHE_LOCK = Lock()


def _get_or_create_cache(skills_dir: str) -> SkillsCache:
    """Get or create a cache for the given skills directory."""
    skills_dir = str(Path(skills_dir).expanduser().resolve())
    with _CACHE_LOCK:
        if skills_dir not in _cache:
            cache = SkillsCache(skills_dir=Path(skills_dir))
            # Auto-discover skills
            if cache.skills_dir.exists():
                cache.skills = _discover_skills(cache.skills_dir)
            _cache[skills_dir] = cache
        return _cache[skills_dir]


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Validate that path is safely contained within base_dir."""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError("Invalid SKILL.md format - missing YAML frontmatter")

    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()

    return frontmatter, body


def _validate_skill_name(name: str) -> bool:
    """Validate skill name follows AgentSkills.io spec (kebab-case, max 64 chars)."""
    if not name or len(name) > 64:
        return False
    return bool(re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", name))


def _discover_skills(skills_dir: Path) -> Dict[str, SkillMetadata]:
    """Discover all skills in a directory (Phase 1 - metadata only)."""
    discovered = {}

    if not skills_dir.exists() or not skills_dir.is_dir():
        return discovered

    for skill_folder in skills_dir.iterdir():
        if not skill_folder.is_dir():
            continue

        if not _is_safe_path(skill_folder, skills_dir):
            logger.warning(f"Skipping unsafe path: {skill_folder}")
            continue

        # Find SKILL.md (case-insensitive)
        skill_md = None
        for f in skill_folder.iterdir():
            if f.name.upper() == "SKILL.MD":
                skill_md = f
                break

        if not skill_md or not _is_safe_path(skill_md, skills_dir):
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _ = _parse_frontmatter(content)

            name = frontmatter.get("name", skill_folder.name)
            description = frontmatter.get("description", "")

            if not _validate_skill_name(name):
                logger.warning(f"Invalid skill name '{name}' in {skill_folder}")
                continue

            discovered[name] = SkillMetadata(
                name=name,
                description=description,
                path=skill_folder,
                license=frontmatter.get("license"),
                allowed_tools=frontmatter.get("allowed-tools"),
            )
            logger.debug(f"Discovered skill: {name}")

        except Exception as e:
            logger.warning(f"Error parsing skill in {skill_folder}: {e}")

    logger.info(f"Discovered {len(discovered)} skills in {skills_dir}")
    return discovered


def _load_skill_instructions(skill_path: Path) -> str:
    """Load full instructions from a skill's SKILL.md (Phase 2)."""
    skill_md = None
    for f in skill_path.iterdir():
        if f.name.upper() == "SKILL.MD":
            skill_md = f
            break

    if not skill_md or not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

    content = skill_md.read_text(encoding="utf-8")
    _, instructions = _parse_frontmatter(content)
    return instructions


def _load_resource(skill_path: Path, resource_path: str) -> str:
    """Load a resource file from a skill (Phase 3)."""
    full_path = skill_path / resource_path

    if not _is_safe_path(full_path, skill_path):
        raise ValueError(f"Invalid resource path: {resource_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    if not full_path.is_file():
        raise ValueError(f"Resource is not a file: {resource_path}")

    if full_path.stat().st_size > MAX_RESOURCE_SIZE:
        raise ValueError(f"Resource too large (max {MAX_RESOURCE_SIZE} bytes): {resource_path}")

    return full_path.read_text(encoding="utf-8")


# Action handlers


def _action_list(skills_dir: str, **kwargs) -> Dict[str, Any]:
    """List all available skills with their descriptions."""
    cache = _get_or_create_cache(skills_dir)

    if not cache.skills:
        return {
            "status": "success",
            "content": [{"text": f"No skills found in {skills_dir}"}],
        }

    lines = [f"Available skills ({len(cache.skills)}):\n"]
    for name, metadata in sorted(cache.skills.items()):
        lines.append(f"  • {name}")
        desc = metadata.description[:100] + "..." if len(metadata.description) > 100 else metadata.description
        lines.append(f"    {desc}")
        if metadata.allowed_tools:
            lines.append(f"    Allowed tools: {', '.join(metadata.allowed_tools)}")

    lines.append("\nUse skills(action='use', skill_name='<name>') to load a skill's instructions.")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_use(skills_dir: str, skill_name: str, **kwargs) -> Dict[str, Any]:
    """Load a skill's full instructions."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}

    cache = _get_or_create_cache(skills_dir)

    if skill_name not in cache.skills:
        available = ", ".join(sorted(cache.skills.keys())) if cache.skills else "none"
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill_name}' not found. Available: {available}"}],
        }

    try:
        metadata = cache.skills[skill_name]
        instructions = _load_skill_instructions(metadata.path)

        result = f"""## Skill: {metadata.name}

{metadata.description}

### Instructions

{instructions}
"""
        return {"status": "success", "content": [{"text": result}]}

    except Exception as e:
        logger.error(f"Error loading skill '{skill_name}': {e}")
        return {"status": "error", "content": [{"text": f"Failed to load skill: {e}"}]}


def _action_get_resource(skills_dir: str, skill_name: str, resource_path: str, **kwargs) -> Dict[str, Any]:
    """Load a resource file from a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}
    if not resource_path:
        return {"status": "error", "content": [{"text": "resource_path is required"}]}

    cache = _get_or_create_cache(skills_dir)

    if skill_name not in cache.skills:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    try:
        metadata = cache.skills[skill_name]
        content = _load_resource(metadata.path, resource_path)

        return {
            "status": "success",
            "content": [{"text": f"# {skill_name}/{resource_path}\n\n```\n{content}\n```"}],
        }

    except Exception as e:
        logger.error(f"Error loading resource: {e}")
        return {"status": "error", "content": [{"text": f"Failed to load resource: {e}"}]}


def _action_list_resources(skills_dir: str, skill_name: str, **kwargs) -> Dict[str, Any]:
    """List available resources in a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}

    cache = _get_or_create_cache(skills_dir)

    if skill_name not in cache.skills:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    metadata = cache.skills[skill_name]
    resources = {"scripts": [], "references": [], "assets": [], "other": []}

    for item in metadata.path.rglob("*"):
        if item.is_file() and item.name.upper() != "SKILL.MD":
            rel_path = item.relative_to(metadata.path)
            # Use Path.parts for cross-platform directory detection
            top_dir = rel_path.parts[0] if len(rel_path.parts) > 1 else None
            # Use as_posix() for consistent display format
            rel_path_str = rel_path.as_posix()

            if top_dir == "scripts":
                resources["scripts"].append(rel_path_str)
            elif top_dir == "references":
                resources["references"].append(rel_path_str)
            elif top_dir == "assets":
                resources["assets"].append(rel_path_str)
            else:
                resources["other"].append(rel_path_str)

    lines = [f"Resources in '{skill_name}':\n"]
    for category, files in resources.items():
        if files:
            lines.append(f"  {category}/")
            for f in sorted(files):
                lines.append(f"    • {f}")

    if not any(resources.values()):
        lines.append("  No resources found.")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_import(skills_dir: str, import_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Import skills from an additional directory."""
    if not import_dir:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "import_dir parameter is required for import action. "
                        "Provide the path to a skills directory to import from."
                    )
                }
            ],
        }

    import_path = Path(import_dir).expanduser().resolve()

    if not import_path.exists() or not import_path.is_dir():
        return {
            "status": "error",
            "content": [{"text": f"Import directory not found or not a directory: {import_dir}"}],
        }

    cache = _get_or_create_cache(skills_dir)
    new_skills = _discover_skills(import_path)

    imported = []
    skipped = []

    for name, metadata in new_skills.items():
        if name in cache.skills:
            skipped.append(name)
            logger.warning(f"Skill '{name}' already exists, skipping import")
        else:
            cache.skills[name] = metadata
            imported.append(name)

    lines = [f"Imported {len(imported)} skill(s) from {import_dir}:"]
    if imported:
        for name in sorted(imported):
            lines.append(f"  + {name}")
    if skipped:
        lines.append(f"\nSkipped {len(skipped)} (already exist): {', '.join(sorted(skipped))}")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


_ACTIONS = {
    "list": _action_list,
    "use": _action_use,
    "get_resource": _action_get_resource,
    "list_resources": _action_list_resources,
    "import": _action_import,
}


def get_skills_prompt(skills_dir: Optional[str] = None) -> str:
    """Generate a skills prompt for injection into agent system prompts.

    This is an optional helper for users who want proactive skill awareness.
    It returns an XML-formatted list of available skills that can be appended
    to your agent's system prompt.

    Uses the same cache as the skills tool, so no duplicate discovery.

    Args:
        skills_dir: Skills directory. Defaults to STRANDS_SKILLS_DIR env var or ./skills

    Returns:
        XML-formatted string with available skills, or empty string if none found.

    Example:
        ```python
        from strands import Agent
        from strands_tools.skills import skills, get_skills_prompt

        # Optional: Add skills to system prompt for proactive awareness
        base_prompt = "You are a helpful assistant."
        agent = Agent(
            tools=[skills],
            system_prompt=base_prompt + get_skills_prompt()
        )
        ```
    """
    skills_dir = skills_dir or os.environ.get("STRANDS_SKILLS_DIR", "./skills")
    cache = _get_or_create_cache(skills_dir)

    if not cache.skills:
        return ""

    lines = [
        "\n\n## Available Skills\n",
        "You have access to specialized skills. Use `skills(action='list')` to see details,",
        "or `skills(action='use', skill_name='<name>')` to load one.\n",
        "<available_skills>",
    ]

    for name, metadata in sorted(cache.skills.items()):
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{metadata.description}</description>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


@tool
def skills(
    action: Literal["list", "use", "get_resource", "list_resources", "import"],
    skill_name: Optional[str] = None,
    resource_path: Optional[str] = None,
    import_dir: Optional[str] = None,
    STRANDS_SKILLS_DIR: Optional[str] = None,
) -> Dict[str, Any]:
    """⚠️ EXPERIMENTAL: This tool is an early experiment for working with Agent Skills. \
The recommended path for production use will be the SDK's native skills feature (coming soon). \
APIs and behavior may change without notice.

    Load and use Agent Skills - modular packages of specialized instructions.

    Skills are folders containing SKILL.md files with instructions that help you
    perform specific tasks effectively. Skills are auto-discovered from STRANDS_SKILLS_DIR.

    Actions:
    - list: Show all available skills with descriptions
    - use: Load a skill's full instructions (returns them for you to follow)
    - get_resource: Load a specific file from a skill (scripts, references, etc.)
    - list_resources: List all files available in a skill
    - import: Dynamically import skills from an additional directory at runtime

    Args:
        action: The action to perform
        skill_name: Name of the skill (required for use, get_resource, list_resources)
        resource_path: Path to resource file (required for get_resource)
        import_dir: Path to directory to import skills from (required for import action)
        STRANDS_SKILLS_DIR: Skills directory. Defaults to STRANDS_SKILLS_DIR env var or ./skills

    Returns:
        Dict with status and content

    Examples:
        # List available skills
        skills(action="list")

        # Load a skill's instructions
        skills(action="use", skill_name="code-reviewer")

        # Get a resource file
        skills(action="get_resource", skill_name="code-reviewer", resource_path="scripts/analyze.py")

        # Import skills from another directory
        skills(action="import", import_dir="/path/to/more/skills")
    """
    warnings.warn(
        "The skills tool is experimental. The recommended path for production use will be "
        "the SDK's native skills feature. APIs and behavior may change without notice.",
        stacklevel=2,
    )

    skills_dir = STRANDS_SKILLS_DIR or os.environ.get("STRANDS_SKILLS_DIR", "./skills")

    if action not in _ACTIONS:
        return {"status": "error", "content": [{"text": f"Invalid action. Valid: {', '.join(_ACTIONS.keys())}"}]}

    try:
        return _ACTIONS[action](
            skills_dir=skills_dir,
            skill_name=skill_name,
            resource_path=resource_path,
            import_dir=import_dir,
        )
    except Exception as e:
        logger.error(f"Error in skills tool: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
