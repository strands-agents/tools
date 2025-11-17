"""
Tests for the cron tool using the Agent interface.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from strands import Agent

from strands_tools import cron
from strands_tools.cron import _sanitize_description


@pytest.fixture
def agent():
    """Create an agent with the cron tool loaded."""
    return Agent(tools=[cron])


@pytest.fixture
def mock_subprocess():
    """Create a mock for subprocess calls."""
    with patch("strands_tools.cron.subprocess") as mock:
        # Set up default responses for run
        mock_result = Mock()
        mock_result.stdout = "0 * * * * echo hello\n30 5 * * * backup.sh\n"
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock.run.return_value = mock_result

        # Mock Popen with a proper context manager
        mock_popen = MagicMock()
        mock_popen.__enter__.return_value.stdin = MagicMock()
        mock.Popen.return_value = mock_popen

        yield mock


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return "\n".join([item["text"] for item in result["content"]])
    return str(result)


def test_list_jobs_success(mock_subprocess, agent):
    """Test listing cron jobs successfully."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n30 5 * * * backup.sh\n"

    # Call through agent
    result = agent.tool.cron(action="list")

    # Verify
    result_text = extract_result_text(result)
    assert "Found 2 cron jobs" in result_text
    assert "ID: 0" in result_text
    assert "echo hello" in result_text
    assert "backup.sh" in result_text
    mock_subprocess.run.assert_called_once_with(["crontab", "-l"], capture_output=True, text=True)


def test_list_jobs_empty(mock_subprocess, agent):
    """Test listing when no cron jobs exist."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = ""

    # Call through agent
    result = agent.tool.cron(action="list")

    # Verify
    result_text = extract_result_text(result)
    assert "No cron jobs found" in result_text


def test_list_jobs_no_crontab(mock_subprocess, agent):
    """Test listing when crontab doesn't exist."""
    # Setup mock to simulate "no crontab for user" error
    mock_subprocess.run.return_value.returncode = 1
    mock_subprocess.run.return_value.stderr = "no crontab for user"
    mock_subprocess.run.return_value.stdout = ""

    # Call through agent
    result = agent.tool.cron(action="list")

    # Verify
    result_text = extract_result_text(result)
    assert "No cron jobs found" in result_text


def test_add_job_success(mock_subprocess, agent):
    """Test adding a cron job successfully."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n"

    # Call through agent
    result = agent.tool.cron(action="add", schedule="30 5 * * *", command="backup.sh", description="Daily backup")

    # Verify
    result_text = extract_result_text(result)
    assert "Successfully added new cron job" in result_text
    assert "30 5 * * * backup.sh" in result_text

    # Check that the crontab was updated
    mock_subprocess.run.assert_called_once()
    mock_subprocess.Popen.assert_called_once()
    # The second argument to write should be the new crontab content
    expected_new_content = "0 * * * * echo hello\n30 5 * * * backup.sh # Daily backup\n"
    mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.assert_called_once_with(expected_new_content)


def test_add_job_missing_params(agent):
    """Test adding a job with missing parameters."""
    # Call without schedule
    result = agent.tool.cron(action="add", command="backup.sh")
    result_text = extract_result_text(result)
    assert "Schedule is required" in result_text

    # Call without command
    result = agent.tool.cron(action="add", schedule="30 5 * * *")
    result_text = extract_result_text(result)
    assert "Command is required" in result_text


def test_raw_entry_success(mock_subprocess, agent):
    """Test adding a raw crontab entry."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n"

    # Call through agent
    raw_entry = "30 5 * * * /bin/bash /path/to/script.sh >> /logs/output.log 2>&1"
    result = agent.tool.cron(action="raw", command=raw_entry)

    # Verify
    result_text = extract_result_text(result)
    assert "Successfully added raw crontab entry" in result_text

    # Check that the crontab was updated
    mock_subprocess.Popen.assert_called_once()
    expected_new_content = f"0 * * * * echo hello\n{raw_entry}\n"
    mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.assert_called_once_with(expected_new_content)


def test_raw_entry_missing_command(agent):
    """Test adding a raw entry without providing the command."""
    result = agent.tool.cron(action="raw")
    result_text = extract_result_text(result)
    assert "Raw crontab entry required" in result_text


def test_remove_job_success(mock_subprocess, agent):
    """Test removing a cron job successfully."""
    # Setup mock with two jobs
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n30 5 * * * backup.sh\n"

    # Call through agent to remove the second job (ID 1)
    result = agent.tool.cron(action="remove", job_id=1)

    # Verify
    result_text = extract_result_text(result)
    assert "Successfully removed cron job" in result_text
    assert "30 5 * * * backup.sh" in result_text

    # Check that the crontab was updated with only the first job
    mock_subprocess.Popen.assert_called_once()
    expected_new_content = "0 * * * * echo hello\n"
    mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.assert_called_once_with(expected_new_content)


def test_remove_job_invalid_id(mock_subprocess, agent):
    """Test removing a job with an invalid ID."""
    # Setup mock with one job
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n"

    # Try to remove job with ID out of range
    result = agent.tool.cron(action="remove", job_id=10)

    # Verify
    result_text = extract_result_text(result)
    assert "Job ID 10 is out of range" in result_text

    # Make sure Popen was not called (crontab was not modified)
    mock_subprocess.Popen.assert_not_called()


def test_remove_job_missing_id(agent):
    """Test removing a job without providing the job ID."""
    result = agent.tool.cron(action="remove")
    result_text = extract_result_text(result)
    assert "Job ID is required" in result_text


def test_edit_job_success(mock_subprocess, agent):
    """Test editing a cron job successfully."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n30 5 * * * backup.sh\n"

    # Call through agent to edit the second job (ID 1)
    result = agent.tool.cron(
        action="edit", job_id=1, schedule="0 3 * * *", command="/usr/bin/backup.sh", description="Updated backup"
    )

    # Verify
    result_text = extract_result_text(result)
    assert "Successfully updated cron job" in result_text
    assert "0 3 * * * /usr/bin/backup.sh # Updated backup" in result_text

    # Check that the crontab was updated
    mock_subprocess.Popen.assert_called_once()
    expected_new_content = "0 * * * * echo hello\n0 3 * * * /usr/bin/backup.sh # Updated backup\n"
    mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.assert_called_once_with(expected_new_content)


def test_edit_job_partial_update(mock_subprocess, agent):
    """Test editing only some fields of a cron job."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n30 5 * * * backup.sh\n"

    # Call through agent to edit only the schedule
    result = agent.tool.cron(action="edit", job_id=1, schedule="0 3 * * *")

    # Verify
    result_text = extract_result_text(result)
    assert "Successfully updated cron job" in result_text
    assert "0 3 * * * backup.sh" in result_text

    # Check that the crontab was updated with the right content
    mock_subprocess.Popen.assert_called_once()
    expected_new_content = "0 * * * * echo hello\n0 3 * * * backup.sh\n"
    mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.assert_called_once_with(expected_new_content)


def test_edit_job_invalid_id(mock_subprocess, agent):
    """Test editing a job with an invalid ID."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = "0 * * * * echo hello\n"

    # Try to edit job with ID out of range
    result = agent.tool.cron(action="edit", job_id=5, schedule="0 3 * * *", command="new_command.sh")

    # Verify
    result_text = extract_result_text(result)
    assert "Job ID 5 is out of range" in result_text

    # Make sure Popen was not called (crontab was not modified)
    mock_subprocess.Popen.assert_not_called()


def test_edit_job_missing_id(agent):
    """Test editing a job without providing the job ID."""
    result = agent.tool.cron(action="edit", schedule="0 3 * * *")
    result_text = extract_result_text(result)
    assert "Job ID is required" in result_text


def test_invalid_action(agent):
    """Test with an invalid action."""
    result = agent.tool.cron(action="invalid_action")
    result_text = extract_result_text(result)
    assert "Unknown action 'invalid_action'" in result_text


def test_edit_job_comment_line(mock_subprocess, agent):
    """Test trying to edit a comment line."""
    # Setup mock with a comment line
    mock_subprocess.run.return_value.stdout = "# This is a comment\n0 * * * * echo hello\n"

    # Try to edit the comment line (ID 0)
    result = agent.tool.cron(action="edit", job_id=0, schedule="0 3 * * *", command="new_command.sh")

    # Verify
    result_text = extract_result_text(result)
    assert "Line 0 is a comment, not a cron job" in result_text

    # Make sure Popen was not called (crontab was not modified)
    mock_subprocess.Popen.assert_not_called()


def test_sanitize_description():
    """Test description sanitization to prevent crontab injection."""
    # Test newline injection
    malicious = "safe comment\n0 * * * * rm -rf /"
    sanitized = _sanitize_description(malicious)
    assert "\n" not in sanitized
    assert sanitized == "safe comment 0 * * * * rm -rf /"

    # Test carriage return injection
    malicious = "safe comment\r0 * * * * rm -rf /"
    sanitized = _sanitize_description(malicious)
    assert "\r" not in sanitized
    assert sanitized == "safe comment 0 * * * * rm -rf /"

    # Test multiple line breaks
    malicious = "line1\n\nline2\r\nline3"
    sanitized = _sanitize_description(malicious)
    assert sanitized == "line1 line2 line3"

    # Test normal description (no change)
    normal = "This is a normal description"
    sanitized = _sanitize_description(normal)
    assert sanitized == normal


def test_add_job_with_malicious_description(mock_subprocess, agent):
    """Test that malicious descriptions are sanitized when adding jobs."""
    # Setup mock
    mock_subprocess.run.return_value.stdout = ""

    # Try to inject a malicious cron job via description
    malicious_desc = "backup job\n0 * * * * rm -rf /"
    result = agent.tool.cron(action="add", schedule="0 2 * * *", command="backup.sh", description=malicious_desc)

    # Verify the job was added but description was sanitized
    result_text = extract_result_text(result)
    assert "Successfully added new cron job" in result_text

    # Check that the written crontab doesn't contain newlines in the comment
    mock_subprocess.Popen.assert_called_once()
    written_content = mock_subprocess.Popen.return_value.__enter__.return_value.stdin.write.call_args[0][0]
    lines = written_content.strip().split("\n")
    assert len(lines) == 1  # Should be only one line
    assert "backup job 0 * * * * rm -rf /" in lines[0]  # Sanitized description
