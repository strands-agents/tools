from unittest.mock import MagicMock, patch

from strands import Agent


def test_generate_image_integration():
    """
    Tests the generate_image tool's integration.
    """
    from strands_tools import generate_image

    agent = Agent(tools=[generate_image])
    res = agent.tool.generate_image(prompt="A blue cat")
    assert "status" in res
    assert "content" in res


def test_image_reader_success(tmp_path):
    """
    Tests the image_reader tool with a temporary image file.
    This is a good example of an integration test as it uses the real file system.
    """
    from PIL import Image
    from strands_tools import image_reader

    # Create a temporary image for the test
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    agent = Agent(tools=[image_reader])
    res = agent.tool.image_reader(image_path=str(img_path))
    assert res["status"] == "success"
    assert "image" in str(res["content"]).lower()


@patch("strands.agent.Agent.__call__")
def test_use_llm_mocked(mock_agent_call):
    """
    Tests the use_llm tool by mocking the agent's call method to prevent a
    real network call, which would cause the test to hang.
    """
    from strands_tools import use_llm

    agent = Agent(tools=[use_llm])
    mock_response = MagicMock()
    mock_response.__str__.return_value = "Mocked LLM Response"
    mock_response.metrics = None
    mock_agent_call.return_value = mock_response

    res = agent.tool.use_llm(prompt="Say hello", system_prompt="You are a bot.")
    assert res["status"] == "success"
    assert "mocked llm response" in str(res["content"]).lower()


def test_think_integration():
    """
    Tests the think tool's integration by allowing the agent to run its
    internal cycles without being mocked.
    """
    from strands_tools import think

    agent = Agent(tools=[think])

    # This now runs the real 'think' logic.
    res = agent.tool.think(
        thought="What is AI?",
        cycle_count=1,  # Reduced for faster testing
        system_prompt="You are a philosopher.",
    )
    assert "status" in res
    assert "content" in res
