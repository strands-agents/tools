from unittest.mock import patch

from strands import Agent
from strands_tools import cron, environment


# Patch the user input function to automatically confirm the action.
@patch("strands_tools.utils.user_input.get_user_input", return_value="y")
def test_environment_set_and_get(mock_get_user_input):
    """
    Test setting and getting an environment variable using the environment tool.
    Mocks user input to bypass the confirmation prompt.
    """
    agent = Agent(tools=[environment])
    var, val = "INTEG_TEST_VAR", "abc123"

    res_set = agent.tool.environment(action="set", name=var, value=val)
    assert res_set["status"] == "success"

    res_get = agent.tool.environment(action="get", name=var)
    assert res_get["status"] == "success"
    assert val in str(res_get["content"]).lower()


def test_cron_list():
    """
    Test the 'list' action of the cron tool.
    Mocks subprocess.run to return a predictable crontab entry without
    accessing the actual system crontab.
    """
    agent = Agent(tools=[cron])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "* * * * * echo hello"
        mock_run.return_value.stderr = ""

        res = agent.tool.cron(action="list")
        assert res["status"] == "success"
        assert "echo hello" in str(res["content"]).lower()
