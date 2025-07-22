"""
Integration tests for browser implementations.
"""

from strands_tools.browser import LocalChromiumBrowser, AgentCoreBrowser
from strands import Agent

from tests_integ.ci_environments import skip_if_github_action


@skip_if_github_action.mark
def test_local_browser():
    local_browser = LocalChromiumBrowser()
    agent = Agent(tools=[local_browser.browser])

    result = agent("""
        1. go to https://smithy.io/2.0/index.html
        2. click on the java section,
        3. open a second session on the same website to click on the rust section
        in one tab and typescript section in another tab,
        4. Save a screenshot of each prefixed with 'local'
        5. If and only if all steps succeed and all 3 screenshots are saved
        RESPOND WITH ONLY "PASS" IF NOT RETURN "FAIL"
    """)

    assert "PASS" in result.message["content"][0]["text"]


@skip_if_github_action.mark
def test_agent_core_browser():
    bedrock_browser = AgentCoreBrowser(region="us-west-2")
    agent = Agent(tools=[bedrock_browser.browser])

    result = agent("""
        1. go to https://smithy.io/2.0/index.html
        2. click on the java section, 
        3. open a second session on the same website to click on the rust section
        in one tab and typescript section in another tab,
        4. Save a screenshot of each prefixed with 'bedrock'
        5. If and only if all steps succeed and all 3 screenshots are saved
        RESPOND WITH ONLY "PASS" IF NOT RETURN "FAIL"
    """)

    assert "PASS" in result.message["content"][0]["text"]
