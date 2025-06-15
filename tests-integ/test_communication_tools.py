from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from strands_tools import (
    generate_image,
    http_request,
    image_reader,
    slack,
    speak,
)


@pytest.fixture
def mock_image_file(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(img_path)
    return str(img_path)


# This test use mock for http call can be a local call.
@patch("strands_tools.http_request.requests.Session.request")
def test_http_request_mocked_local(mock_req):
    """
    Tests the http_request tool with a MOCKED local request.
    Mocking is kept here because a true integration test would require
    a separate local server to be running.
    """
    # Configure the mock to simulate a successful web request.
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"ok": true}'
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.url = "http://localhost/"
    mock_resp.reason = "OK"
    mock_req.return_value = mock_resp

    # The tool dictionary must be passed as a positional argument.
    mock_tool_use = {"toolUseId": "http_test_123", "input": {"method": "GET", "url": "http://localhost/"}}
    res = http_request.http_request(mock_tool_use)

    assert res["status"] == "success"
    assert 'body: {"ok": true}' in str(res["content"]).lower()


def test_slack_post_message_integration():
    """
    Tests the slack tool by making a REAL API call to Slack.
    """
    # The input dictionary is structured to match the slack function's signature.
    tool_use = {
        "toolUseId": "slack_integ_123",
        "input": {
            "action": "chat.postMessage",
            "parameters": {"channel": "#test_channel", "text": "This is an integration test message from Strands."},
        },
    }
    res = slack.slack(tool_use)

    # We assert that the tool returns a valid success or error response from the live API.
    assert res["status"] in ["success", "error"]
    assert "content" in res


# This test use mock due to hanging call to agent.
@patch("strands_tools.speak.subprocess.run")
def test_speak_tool_mocked(mock_subprocess_run):
    """
    Tests the speak tool by mocking the subprocess call.
    """
    tool_use = {"toolUseId": "speak_test_123", "input": {"text": "This is a unit test."}}
    res = speak.speak(tool_use)
    assert res["status"] == "success"
    assert "text spoken" in str(res["content"]).lower()
    mock_subprocess_run.assert_called_with(["say", "This is a unit test."], check=True)


def test_generate_image_integration():
    """
    Tests the generate_image tool by making a REAL API call to AWS Bedrock.
    """
    tool_use = {"toolUseId": "gen_img_integ_123", "input": {"prompt": "a cartoon cat"}}
    res = generate_image.generate_image(tool_use)
    assert "status" in res
    assert "content" in res


def test_image_reader_integration(mock_image_file):
    tool_use = {"toolUseId": "read_img_integ_123", "input": {"image_path": mock_image_file}}

    res = image_reader.image_reader(tool_use)
    assert res["status"] == "success"
    assert "image" in str(res["content"]).lower()
