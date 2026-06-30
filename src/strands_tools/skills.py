"""Agent Skills Tool for Strands Agents.

This tool provides a high-level interface for working with Agent Skills -
modular packages of instructions, scripts, and resources that give AI agents
specialized capabilities for specific tasks.

Note: The Strands SDK also provides a native AgentSkills plugin
(strands.vended_plugins.skills.AgentSkills) which offers automatic system
prompt injection and lifecycle management. This tool is a standalone
alternative that works through the tool-calling interface and can be used
independently or alongside the SDK plugin.

Skills follow the AgentSkills.io specification and use progressive disclosure:
- Phase 1: Load only metadata (~100 tokens per skill) via 'list' action
- Phase 2: Load full instructions when skill is used via 'use' action
- Phase 3: Load resources (scripts, references, assets) as needed via 'get_resource'

Environment Variables:
    STRANDS_SKILLS_DIR: Comma-separated list of directories containing skills (default: ./skills)

Usage Examples:
--------------
Using this tool directly:

```python
from strands import Agent
from strands_tools import skills

# Skills are auto-discovered at import time if STRANDS_SKILLS_DIR is set
agent = Agent(tools=[skills])
```

Or call `sync_skills` explicitly before creating the agent (e.g. after fetching skills dynamically):

```python
from strands import Agent
from strands_tools import skills
from strands_tools.skills import sync_skills

sync_skills(skills_dir="./my-skills,~/.agents/skills")
agent = Agent(tools=[skills])
```

Using the SDK's native AgentSkills plugin instead:

```python
from strands import Agent
from strands.vended_plugins.skills import AgentSkills

agent = Agent(plugins=[AgentSkills(skills="./skills")])
```

For more information about Agent Skills:
- Specification: https://agentskills.io
- Anthropic Skills: https://github.com/anthropics/skills
- SDK Plugin: https://strandsagents.com/latest/user-guide/concepts/agent-skills/
"""

import logging
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from strands import tool
from strands.types.tools import ToolContext
from strands.vended_plugins.skills.skill import Skill

logger = logging.getLogger(__name__)

# Maximum file size for resources (10MB)
MAX_RESOURCE_SIZE = 10 * 1024 * 1024

# Maximum size for a single SKILL.md file imported via file path or URL (1MB)
MAX_SKILL_FILE_SIZE = 1 * 1024 * 1024

# Timeout for fetching a SKILL.md from a URL (seconds)
URL_FETCH_TIMEOUT = 30

# Module-level skills cache (matches pattern used by other tools in this package)
_cache: Dict[str, Skill] = {}

# Resource directories to scan (matches the plugin)
_RESOURCE_DIRS = ("scripts", "references", "assets")

# Maximum number of resource files to list in skill responses
_DEFAULT_MAX_RESOURCE_FILES = 20


def _resolve_skills(sources: List[str]) -> Dict[str, Skill]:
    """Resolve a list of skill source paths into Skill instances.

    Each source can be:
    - A path to a skill directory (containing SKILL.md)
    - A path to a parent directory (containing skill subdirectories)
    - A path to a SKILL.md file directly

    Matches the SDK plugin's resolution logic.

    Args:
        sources: List of path strings to resolve.

    Returns:
        Dict mapping skill names to Skill instances.
    """
    resolved: Dict[str, Skill] = {}

    for source in sources:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            logger.warning("path=<%s> | skill source path does not exist, skipping", path)
            continue

        if path.is_dir():
            has_skill_md = (path / "SKILL.md").is_file() or (path / "skill.md").is_file()

            if has_skill_md:
                try:
                    skill = Skill.from_file(path)
                    if skill.name in resolved:
                        logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                    resolved[skill.name] = skill
                except (ValueError, FileNotFoundError) as e:
                    logger.warning("path=<%s> | failed to load skill: %s", path, e)
            else:
                try:
                    for skill in Skill.from_directory(path):
                        if skill.name in resolved:
                            logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                        resolved[skill.name] = skill
                except FileNotFoundError:
                    logger.warning("path=<%s> | skills directory not found, skipping", path)

        elif path.is_file() and path.name.lower() == "skill.md":
            try:
                skill = Skill.from_file(path)
                if skill.name in resolved:
                    logger.warning("name=<%s> | duplicate skill name, overwriting previous skill", skill.name)
                resolved[skill.name] = skill
            except (ValueError, FileNotFoundError) as e:
                logger.warning("path=<%s> | failed to load skill: %s", path, e)

    logger.debug("source_count=<%d>, resolved_count=<%d> | skills resolved", len(sources), len(resolved))
    return resolved


def _sync_skills(skills_dir: str) -> Dict[str, Skill]:
    """Synchronize the skills cache by discovering from all configured paths.

    Splits skills_dir on commas and resolves each path. If the cache is empty,
    populates it from the discovered skills.

    Args:
        skills_dir: Comma-separated list of skill source paths.

    Returns:
        The skills cache dict.
    """
    if not _cache:
        sources = [s.strip() for s in skills_dir.split(",") if s.strip()]
        discovered = _resolve_skills(sources)
        _cache.update(discovered)
    return _cache


def _update_tool_spec(cache: Dict[str, Skill]) -> None:
    """Update the skills tool's description to include the available skills catalog.

    Directly updates the module-level `skills` DecoratedFunctionTool's tool_spec.

    Args:
        cache: The current skills cache.
    """
    skills_xml = _generate_skills_xml(cache)

    base_description = (
        "Load and use Agent Skills - modular packages of specialized instructions.\n\n"
        "Skills are folders containing SKILL.md files with instructions that help you "
        "perform specific tasks effectively. Skills are auto-discovered from STRANDS_SKILLS_DIR.\n\n"
        "Actions:\n"
        "- list: Show all available skills with descriptions\n"
        "- use: Activate a skill and load its full instructions (returns them for you to follow)\n"
        "- get_resource: Load a specific file from a skill (scripts, references, etc.)\n"
        "- list_resources: List all files available in a skill\n"
        "- import: Import skills from a directory, file path, or HTTPS URL\n\n"
        "When a task matches a skill's description, call this tool with action='use' "
        "and the skill's name to load its full instructions."
    )

    full_description = f"{base_description}\n\n{skills_xml}"

    try:
        current_spec = skills.tool_spec
        skills.tool_spec = {
            "name": current_spec["name"],
            "description": full_description,
            "inputSchema": current_spec["inputSchema"],
        }
    except (AttributeError, KeyError, ValueError) as e:
        logger.debug("Failed to update tool spec: %s", e)


def _generate_skills_xml(cache: Dict[str, Skill]) -> str:
    """Generate the XML block listing available skills.

    Matches the SDK plugin's _generate_skills_xml format.

    Args:
        cache: The skills cache dict.

    Returns:
        XML-formatted string with skill metadata.
    """
    if not cache:
        return "<available_skills>\nNo skills are currently available.\n</available_skills>"

    lines: List[str] = ["<available_skills>"]

    for skill in sorted(cache.values(), key=lambda s: s.name):
        lines.append("<skill>")
        lines.append(f"<name>{escape(skill.name)}</name>")
        lines.append(f"<description>{escape(skill.description)}</description>")
        if skill.path is not None:
            lines.append(f"<location>{escape(str(skill.path / 'SKILL.md'))}</location>")
        lines.append("</skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)


def _list_skill_resources(skill_path: Path) -> List[str]:
    """List resource files in a skill's optional directories.

    Scans scripts/, references/, and assets/ subdirectories for files.
    Results are capped at _DEFAULT_MAX_RESOURCE_FILES.

    Args:
        skill_path: Path to the skill directory.

    Returns:
        List of relative file paths.
    """
    files: List[str] = []

    for dir_name in _RESOURCE_DIRS:
        resource_dir = skill_path / dir_name
        if not resource_dir.is_dir():
            continue

        for file_path in sorted(resource_dir.rglob("*")):
            if not file_path.is_file():
                continue
            files.append(file_path.relative_to(skill_path).as_posix())
            if len(files) >= _DEFAULT_MAX_RESOURCE_FILES:
                files.append(f"... (truncated at {_DEFAULT_MAX_RESOURCE_FILES} files)")
                return files

    return files


def _format_skill_response(skill: Skill) -> str:
    """Format the tool response when a skill is activated.

    Includes full instructions, metadata, and resource listing.
    Matches the SDK plugin's _format_skill_response format.

    Args:
        skill: The activated skill.

    Returns:
        Formatted string with skill instructions and metadata.
    """
    if not skill.instructions:
        return f"Skill '{skill.name}' activated (no instructions available)."

    parts: List[str] = [skill.instructions]

    metadata_lines: List[str] = []
    if skill.allowed_tools:
        metadata_lines.append(f"Allowed tools: {', '.join(skill.allowed_tools)}")
    if skill.compatibility:
        metadata_lines.append(f"Compatibility: {skill.compatibility}")
    if skill.path is not None:
        metadata_lines.append(f"Location: {skill.path / 'SKILL.md'}")

    if metadata_lines:
        parts.append("\n---\n" + "\n".join(metadata_lines))

    if skill.path is not None:
        resources = _list_skill_resources(skill.path)
        if resources:
            parts.append("\nAvailable resources:\n" + "\n".join(f"  {r}" for r in resources))

    return "\n".join(parts)


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Validate that path is safely contained within base_dir."""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


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


def _action_list(skills_dir: str, **kwargs: Any) -> Dict[str, Any]:
    """List all available skills with their descriptions."""
    cache = _sync_skills(skills_dir)

    if not cache:
        return {
            "status": "success",
            "content": [{"text": f"No skills found in {skills_dir}"}],
        }

    lines = [f"Available skills ({len(cache)}):\n"]
    for name, skill in sorted(cache.items()):
        lines.append(f"  • {name}")
        desc = skill.description[:100] + "..." if len(skill.description) > 100 else skill.description
        lines.append(f"    {desc}")
        if skill.allowed_tools:
            lines.append(f"    Allowed tools: {', '.join(skill.allowed_tools)}")

    lines.append("\nUse skills(action='use', skill_name='<name>') to load a skill's instructions.")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_use(skills_dir: str, skill_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Activate a skill and load its full instructions."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}

    cache = _sync_skills(skills_dir)

    if skill_name not in cache:
        available = ", ".join(sorted(cache.keys())) if cache else "none"
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill_name}' not found. Available: {available}"}],
        }

    try:
        skill = cache[skill_name]
        result = _format_skill_response(skill)
        return {"status": "success", "content": [{"text": result}]}

    except Exception as e:
        logger.error(f"Error loading skill '{skill_name}': {e}")
        return {"status": "error", "content": [{"text": f"Failed to load skill: {e}"}]}


def _action_get_resource(skills_dir: str, skill_name: str, resource_path: str, **kwargs: Any) -> Dict[str, Any]:
    """Load a resource file from a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}
    if not resource_path:
        return {"status": "error", "content": [{"text": "resource_path is required"}]}

    cache = _sync_skills(skills_dir)

    if skill_name not in cache:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    skill = cache[skill_name]

    if skill.path is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"Skill '{skill_name}' was imported from a URL "
                        "and has no local resources. Only directory-based skills support get_resource."
                    )
                }
            ],
        }

    try:
        content = _load_resource(skill.path, resource_path)

        return {
            "status": "success",
            "content": [{"text": f"# {skill_name}/{resource_path}\n\n```\n{content}\n```"}],
        }

    except Exception as e:
        logger.error(f"Error loading resource: {e}")
        return {"status": "error", "content": [{"text": f"Failed to load resource: {e}"}]}


def _action_list_resources(skills_dir: str, skill_name: str, **kwargs: Any) -> Dict[str, Any]:
    """List available resources in a skill."""
    if not skill_name:
        return {"status": "error", "content": [{"text": "skill_name is required"}]}

    cache = _sync_skills(skills_dir)

    if skill_name not in cache:
        return {"status": "error", "content": [{"text": f"Skill '{skill_name}' not found"}]}

    skill = cache[skill_name]

    if skill.path is None:
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Skill '{skill_name}' was imported from a URL.\n"
                        "No local resources available. Only the skill instructions are loaded."
                    )
                }
            ],
        }

    resources = {"scripts": [], "references": [], "assets": [], "other": []}

    for item in skill.path.rglob("*"):
        if item.is_file() and item.name.upper() != "SKILL.MD":
            rel_path = item.relative_to(skill.path)
            top_dir = rel_path.parts[0] if len(rel_path.parts) > 1 else None
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


def _import_from_content(content: str, source_label: str, cache: Dict[str, Skill]) -> Dict[str, Any]:
    """Parse SKILL.md content and register a virtual skill in the cache.

    Used for URL imports where there is no local directory. The skill is
    marked as virtual (path=None), meaning get_resource and list_resources won't work.

    Args:
        content: Raw SKILL.md text (frontmatter + body).
        source_label: Human-readable origin (file path or URL) for logging/messages.
        cache: The skills cache dict to add the skill to.

    Returns:
        Standard success/error dict.
    """
    try:
        skill = Skill.from_content(content)
    except ValueError as e:
        return {
            "status": "error",
            "content": [{"text": f"Invalid SKILL.md format from {source_label}: {e}"}],
        }

    if skill.name in cache:
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill.name}' already exists, cannot import from {source_label}"}],
        }

    # Virtual skill — no local path
    skill.path = None
    cache[skill.name] = skill

    logger.info(f"Imported virtual skill '{skill.name}' from {source_label}")
    return {
        "status": "success",
        "content": [{"text": f"Imported skill '{skill.name}' from {source_label}"}],
    }


def _import_from_file(source: str, cache: Dict[str, Skill]) -> Dict[str, Any]:
    """Import a single skill from a local SKILL.md file path.

    Args:
        source: Path to a SKILL.md file (may contain ~ or relative segments).
        cache: The skills cache dict to add the skill to.

    Returns:
        Standard success/error dict.
    """
    file_path = Path(source).expanduser().resolve()

    if not file_path.exists():
        return {
            "status": "error",
            "content": [{"text": f"Source file not found: {source}"}],
        }
    if not file_path.is_file():
        return {
            "status": "error",
            "content": [{"text": f"Source is not a file: {source}"}],
        }
    if file_path.stat().st_size > MAX_SKILL_FILE_SIZE:
        return {
            "status": "error",
            "content": [{"text": f"Source file too large (max {MAX_SKILL_FILE_SIZE} bytes): {source}"}],
        }

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "status": "error",
            "content": [{"text": f"Source file is not valid UTF-8: {source}"}],
        }
    except OSError as e:
        return {
            "status": "error",
            "content": [{"text": f"Error reading source file {source}: {e}"}],
        }

    try:
        skill = Skill.from_content(content)
    except ValueError as e:
        return {
            "status": "error",
            "content": [{"text": f"Invalid SKILL.md format from {source}: {e}"}],
        }

    # Set path to the parent directory so resources are accessible
    skill.path = file_path.parent

    if skill.name in cache:
        return {
            "status": "error",
            "content": [{"text": f"Skill '{skill.name}' already exists, cannot import from {source}"}],
        }

    cache[skill.name] = skill

    logger.info(f"Imported skill '{skill.name}' from {file_path}")
    return {
        "status": "success",
        "content": [{"text": f"Imported skill '{skill.name}' from {source}"}],
    }


def _import_from_url(source: str, cache: Dict[str, Skill]) -> Dict[str, Any]:
    """Import a single skill from an HTTPS URL pointing to a SKILL.md file.

    Args:
        source: HTTPS URL to a SKILL.md file.
        cache: The skills cache dict to add the skill to.

    Returns:
        Standard success/error dict.
    """
    parsed = urlparse(source)
    if parsed.scheme != "https":
        return {
            "status": "error",
            "content": [{"text": "Only HTTPS URLs are supported for skill import"}],
        }
    if not parsed.hostname:
        return {
            "status": "error",
            "content": [{"text": f"Invalid URL (no hostname): {source}"}],
        }

    try:
        req = urllib.request.Request(
            source,
            headers={"User-Agent": "strands-agents-skills/1.0"},
        )
        ssl_context = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=URL_FETCH_TIMEOUT, context=ssl_context) as resp:
            # Guard against redirect to non-HTTPS
            if hasattr(resp, "url") and resp.url and not resp.url.startswith("https://"):
                return {
                    "status": "error",
                    "content": [{"text": f"URL redirected to non-HTTPS location: {resp.url}"}],
                }

            data = resp.read(MAX_SKILL_FILE_SIZE + 1)
            if len(data) > MAX_SKILL_FILE_SIZE:
                return {
                    "status": "error",
                    "content": [{"text": f"URL content too large (max {MAX_SKILL_FILE_SIZE} bytes)"}],
                }

        content = data.decode("utf-8")

    except urllib.error.HTTPError as e:
        return {
            "status": "error",
            "content": [{"text": f"HTTP error fetching URL: {e.code} {e.reason}"}],
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "content": [{"text": f"Error fetching URL: {e.reason}"}],
        }
    except TimeoutError as e:
        return {
            "status": "error",
            "content": [{"text": f"Timeout fetching URL (limit {URL_FETCH_TIMEOUT}s): {e}"}],
        }
    except UnicodeDecodeError:
        return {
            "status": "error",
            "content": [{"text": "URL content is not valid UTF-8"}],
        }
    except OSError as e:
        return {
            "status": "error",
            "content": [{"text": f"Network error fetching URL: {e}"}],
        }

    return _import_from_content(content, source_label=source, cache=cache)


def _import_from_directory(source: str, cache: Dict[str, Skill]) -> Dict[str, Any]:
    """Import skills from a directory.

    Handles two cases:
    - If the directory itself contains SKILL.md, imports it as a single skill
    - Otherwise, scans child subdirectories for SKILL.md files

    Args:
        source: Path to a skill directory or a parent directory containing skill folders.
        cache: The skills cache dict to add skills to.

    Returns:
        Standard success/error dict.
    """
    import_path = Path(source).expanduser().resolve()

    if not import_path.exists() or not import_path.is_dir():
        return {
            "status": "error",
            "content": [{"text": f"Import directory not found or not a directory: {source}"}],
        }

    # Check if the directory itself is a skill (contains SKILL.md)
    has_skill_md = (import_path / "SKILL.md").is_file() or (import_path / "skill.md").is_file()
    if has_skill_md:
        skill_file = "SKILL.md" if (import_path / "SKILL.md").is_file() else "skill.md"
        return _import_from_file(str(import_path / skill_file), cache)

    try:
        new_skills = Skill.from_directory(import_path)
    except FileNotFoundError:
        return {
            "status": "error",
            "content": [{"text": f"Import directory not found: {source}"}],
        }

    imported = []
    skipped = []

    for skill in new_skills:
        if skill.name in cache:
            skipped.append(skill.name)
            logger.warning(f"Skill '{skill.name}' already exists, skipping import")
        else:
            cache[skill.name] = skill
            imported.append(skill.name)

    lines = [f"Imported {len(imported)} skill(s) from {source}:"]
    if imported:
        for name in sorted(imported):
            lines.append(f"  + {name}")
    if skipped:
        lines.append(f"\nSkipped {len(skipped)} (already exist): {', '.join(sorted(skipped))}")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_import(skills_dir: str, source: str = None, **kwargs: Any) -> Dict[str, Any]:
    """Import skills from a directory, file path, or URL.

    Args:
        skills_dir: The main skills directory (for cache lookup).
        source: Path to a directory, SKILL.md file, or HTTPS URL.

    Returns:
        Standard success/error dict.
    """
    if not source:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "source parameter is required for import action. "
                        "Provide a path to a skills directory, a SKILL.md file, or an HTTPS URL."
                    )
                }
            ],
        }

    cache = _sync_skills(skills_dir)

    # HTTPS URL
    if source.startswith("https://"):
        return _import_from_url(source, cache)

    # Reject non-HTTPS URL schemes explicitly
    if source.startswith(("http://", "ftp://", "file://")):
        return {
            "status": "error",
            "content": [{"text": "Only HTTPS URLs are supported for skill import"}],
        }

    # Local path — determine if it's a file or directory
    local_path = Path(source).expanduser().resolve()

    if local_path.is_dir():
        return _import_from_directory(source, cache)

    if local_path.is_file():
        return _import_from_file(source, cache)

    return {
        "status": "error",
        "content": [{"text": f"Source not found: {source}"}],
    }


def sync_skills(skills_dir: Optional[str] = None) -> None:
    """Pre-load skills into the cache and update the tool spec.

    Call this before or after creating an Agent to eagerly discover skills and
    populate the tool description with the skills catalog, so the model sees
    available skills from the first turn.

    Args:
        skills_dir: Comma-separated skills directories.
            Defaults to STRANDS_SKILLS_DIR env var or ./skills.

    Example::

        from strands import Agent
        from strands_tools import skills
        from strands_tools.skills import sync_skills

        sync_skills(skills_dir="./my-skills,~/.agents/skills")
        agent = Agent(tools=[skills])
    """
    resolved_dir = skills_dir or os.environ.get("STRANDS_SKILLS_DIR", "./skills")
    sources = [s.strip() for s in resolved_dir.split(",") if s.strip()]
    discovered = _resolve_skills(sources)
    _cache.update(discovered)
    _update_tool_spec(_cache)


_ACTIONS = {
    "list": _action_list,
    "use": _action_use,
    "get_resource": _action_get_resource,
    "list_resources": _action_list_resources,
    "import": _action_import,
}


@tool(context=True)
def skills(
    tool_context: ToolContext,
    action: Literal["list", "use", "get_resource", "list_resources", "import"],
    skill_name: Optional[str] = None,
    resource_path: Optional[str] = None,
    source: Optional[str] = None,
    skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and use Agent Skills - modular packages of specialized instructions.

    Skills are folders containing SKILL.md files with instructions that help you
    perform specific tasks effectively. Skills are auto-discovered from STRANDS_SKILLS_DIR.

    Actions:
    - list: Show all available skills with descriptions
    - use: Activate a skill and load its full instructions (returns them for you to follow)
    - get_resource: Load a specific file from a skill (scripts, references, etc.)
    - list_resources: List all files available in a skill
    - import: Import skills from a directory, file path, or HTTPS URL

    When a task matches a skill's description, call this tool with action='use'
    and the skill's name to load its full instructions.

    Args:
        tool_context: The tool context (automatically injected)
        action: The action to perform
        skill_name: Name of the skill (required for use, get_resource, list_resources)
        resource_path: Path to resource file (required for get_resource)
        source: Source to import from (required for import action). Accepts:
            - Directory path: imports all skills from the directory
            - File path: imports a single skill from a SKILL.md file
            - HTTPS URL: imports a single skill from a remote SKILL.md
        skills_dir: Comma-separated skills directories. Defaults to STRANDS_SKILLS_DIR env var or ./skills

    Returns:
        Dict with status and content

    Examples:
        # List available skills
        skills(action="list")

        # Load a skill's instructions
        skills(action="use", skill_name="code-reviewer")

        # Get a resource file
        skills(action="get_resource", skill_name="code-reviewer", resource_path="scripts/analyze.py")

        # Import skills from a directory
        skills(action="import", source="/path/to/more/skills")

        # Import a single skill from a file
        skills(action="import", source="/path/to/my-skill/SKILL.md")

        # Import a skill from a URL
        skills(action="import", source="https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md")
    """
    skills_dir = skills_dir or os.environ.get("STRANDS_SKILLS_DIR", "./skills")

    if action not in _ACTIONS:
        return {"status": "error", "content": [{"text": f"Invalid action. Valid: {', '.join(_ACTIONS.keys())}"}]}

    try:
        result = _ACTIONS[action](
            tool_context=tool_context,
            skills_dir=skills_dir,
            skill_name=skill_name,
            resource_path=resource_path,
            source=source,
        )

        # Update tool spec after action so imports are reflected immediately
        _update_tool_spec(_cache)

        return result
    except Exception as e:
        logger.error(f"Error in skills tool: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}


# Auto-discover skills at import time if STRANDS_SKILLS_DIR is set
if os.environ.get("STRANDS_SKILLS_DIR"):
    sync_skills()
