import json
import re

from strands import Agent
from strands_tools import calculator, current_time, stop


def test_calculator_operations():
    """Test calculator tool for basic math operations."""

    agent = Agent(tools=[calculator])

    res = agent.tool.calculator(expression="5 + 7 * 2")
    assert res["status"] == "success"
    assert "19" in json.dumps(res["content"])


def test_current_time():
    """Test current_time tool returns valid time information."""

    agent = Agent(tools=[current_time])

    res = agent.tool.current_time()
    assert res["status"] == "success"

    content_str = json.dumps(res["content"]).lower()
    has_year = re.search(r"\d{4}", content_str)
    has_time = re.search(r"\d{1,2}:\d{2}", content_str)

    assert has_year and has_time, f"Expected a date and time in the response, but got: {content_str}"


def test_stop_sets_flag():
    agent = Agent(tools=[stop])
    request_state = {}
    res = agent.tool.stop(reason="integration test", request_state=request_state)
    assert res["status"] == "success"
    assert request_state.get("stop_event_loop") is True
