"""
Crontab manager for scheduling tasks, with special support for Strands agent jobs.

Simple, direct interface to the system's crontab with helpful guidance in documentation.
"""

import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional

from strands import tool

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)


def _sanitize_cron_line(line: str) -> str:
    """Collapse newlines in a crontab line to prevent injection of extra entries."""
    return re.sub(r"[\r\n]+", " ", line).strip()


def _read_crontab() -> str:
    """Read the current crontab content.

    Returns an empty string if no crontab exists for the user.
    """
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0 and "no crontab for" not in result.stderr:
        raise Exception(f"Failed to read crontab: {result.stderr}")
    return result.stdout if result.returncode == 0 else ""


def _write_crontab(new_content: str, description: str) -> Optional[Dict[str, Any]]:
    """Prompt user for consent, then write new crontab content.

    Args:
        new_content: The full crontab content to write.
        description: What's changing (shown in the consent prompt).

    Returns:
        An error dict if the user denies consent, None on success.
    """
    bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    if not bypass_consent:
        console = console_util.create()
        console.print(f"\n[bold yellow]Cron tool wants to modify your crontab:[/bold yellow]\n{description}\n")
        confirm = get_user_input("<yellow><bold>Allow this crontab modification?</bold> [y/*]</yellow>")

        if confirm.lower() != "y":
            return {
                "status": "error",
                "content": [{"text": f"Crontab modification cancelled by user. Input: {confirm}"}],
            }

    with subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True) as proc:
        proc.stdin.write(new_content)

    return None


@tool
def cron(
    action: str,
    schedule: Optional[str] = None,
    command: Optional[str] = None,
    job_id: Optional[int] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Manage crontab entries for scheduling tasks, with special support for Strands agent jobs.

    This tool provides full access to your system's crontab while offering helpful patterns
    and best practices for Strands agent scheduling.

    # Strands Agent Job Best Practices:
    - Use 'BYPASS_TOOL_CONSENT=true strands "<your_prompt>"' to run Strands agent tasks
    - Always add output redirection to log files: '>> /path/to/log.file 2>&1'
    - Example: 'BYPASS_TOOL_CONSENT=true strands "Generate a report" >> /tmp/report.log 2>&1'
    - Consider creating organized log directories like '/tmp/strands_logs/'

    # Cron Schedule Examples:
    - Every 5 minutes: '*/5 * * * *'
    - Daily at 8 AM: '0 8 * * *'
    - Every Monday at noon: '0 12 * * 1'
    - First day of month: '0 0 1 * *'

    Args:
        action: Action to perform. Must be one of: 'list', 'add', 'remove', 'edit', 'raw'
            - 'raw': Directly edit crontab with specified raw cron entry (use with command parameter)
        schedule: Cron schedule expression (e.g., '*/5 * * * *' for every 5 minutes)
        command: The command to schedule in crontab
        job_id: ID of the job to remove or edit (line number in crontab)
        description: Optional description for this cron job (added as comment)

    Returns:
        Dict containing status and response content
    """
    try:
        if action.lower() == "list":
            return list_jobs()
        elif action.lower() == "add":
            if not schedule:
                return {"status": "error", "content": [{"text": "Error: Schedule is required"}]}
            if not command:
                return {"status": "error", "content": [{"text": "Error: Command is required"}]}
            return add_job(schedule, command, description)
        elif action.lower() == "raw":
            if not command:
                return {"status": "error", "content": [{"text": "Error: Raw crontab entry required"}]}
            return add_raw_entry(command)
        elif action.lower() == "remove":
            if job_id is None:
                return {"status": "error", "content": [{"text": "Error: Job ID is required"}]}
            return remove_job(job_id)
        elif action.lower() == "edit":
            if job_id is None:
                return {"status": "error", "content": [{"text": "Error: Job ID is required"}]}
            return edit_job(job_id, schedule, command, description)
        else:
            return {"status": "error", "content": [{"text": f"Error: Unknown action '{action}'"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def list_jobs() -> Dict[str, Any]:
    """List all cron jobs in the crontab."""
    try:
        crontab = _read_crontab()

        jobs = []
        for i, line in enumerate(crontab.splitlines()):
            line = line.strip()
            if line and not line.startswith("#"):
                jobs.append({"id": i, "line": line})

        if jobs:
            content = [{"text": f"Found {len(jobs)} cron jobs:"}]
            for job in jobs:
                content.append({"text": f"ID: {job['id']}\n{job['line']}"})
        else:
            content = [{"text": "No cron jobs found in crontab"}]

        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error listing cron jobs: {str(e)}"}]}


def add_job(schedule: str, command: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Add a new cron job to the crontab."""
    try:
        crontab = _read_crontab()

        description_text = f"# {description}" if description else ""
        cron_line = _sanitize_cron_line(f"{schedule} {command} {description_text}")

        new_crontab = crontab.rstrip() + "\n" + cron_line + "\n" if crontab else cron_line + "\n"

        denial = _write_crontab(new_crontab, f"Add job: {cron_line}")
        if denial:
            return denial

        return {"status": "success", "content": [{"text": f"Successfully added new cron job: {cron_line}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error adding cron job: {str(e)}"}]}


def add_raw_entry(raw_entry: str) -> Dict[str, Any]:
    """Add a raw crontab entry directly to the crontab."""
    try:
        crontab = _read_crontab()

        sanitized_entry = _sanitize_cron_line(raw_entry)

        new_crontab = crontab.rstrip() + "\n" + sanitized_entry + "\n" if crontab else sanitized_entry + "\n"

        denial = _write_crontab(new_crontab, f"Add raw entry: {sanitized_entry}")
        if denial:
            return denial

        return {"status": "success", "content": [{"text": f"Successfully added raw crontab entry: {sanitized_entry}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error adding raw crontab entry: {str(e)}"}]}


def remove_job(job_id: int) -> Dict[str, Any]:
    """Remove a cron job from the crontab by ID (line number)."""
    try:
        crontab = _read_crontab()
        crontab_lines = crontab.splitlines()

        if job_id < 0 or job_id >= len(crontab_lines):
            return {"status": "error", "content": [{"text": f"Error: Job ID {job_id} is out of range"}]}

        removed_job = crontab_lines.pop(job_id)
        new_crontab = "\n".join(crontab_lines) + "\n" if crontab_lines else ""

        denial = _write_crontab(new_crontab, f"Remove job: {removed_job}")
        if denial:
            return denial

        return {"status": "success", "content": [{"text": f"Successfully removed cron job: {removed_job}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error removing cron job: {str(e)}"}]}


def edit_job(
    job_id: int, schedule: Optional[str], command: Optional[str], description: Optional[str]
) -> Dict[str, Any]:
    """Edit an existing cron job in the crontab."""
    try:
        crontab = _read_crontab()
        crontab_lines = crontab.splitlines()

        if job_id < 0 or job_id >= len(crontab_lines):
            return {"status": "error", "content": [{"text": f"Error: Job ID {job_id} is out of range"}]}

        old_line = crontab_lines[job_id].strip()

        if old_line.startswith("#"):
            return {"status": "error", "content": [{"text": f"Error: Line {job_id} is a comment, not a cron job"}]}

        # Parse existing job (first 5 space-separated segments form the schedule)
        parts = old_line.split(None, 5)
        if len(parts) < 6:
            return {"status": "error", "content": [{"text": "Error: Invalid cron format"}]}

        old_schedule = " ".join(parts[:5])
        old_command_rest = parts[5]

        comment_idx = old_command_rest.find("#")
        old_command = old_command_rest
        old_comment = ""

        if comment_idx >= 0:
            old_command = old_command_rest[:comment_idx].strip()
            old_comment = old_command_rest[comment_idx:].strip()

        new_schedule = schedule if schedule is not None else old_schedule
        new_command = command if command is not None else old_command
        new_comment = f"# {description}" if description is not None else old_comment

        new_cron_line = _sanitize_cron_line(f"{new_schedule} {new_command} {new_comment}")

        crontab_lines[job_id] = new_cron_line
        new_crontab = "\n".join(crontab_lines) + "\n"

        denial = _write_crontab(new_crontab, f"Edit job {job_id}: {new_cron_line}")
        if denial:
            return denial

        return {"status": "success", "content": [{"text": f"Successfully updated cron job to: {new_cron_line}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error editing cron job: {str(e)}"}]}
