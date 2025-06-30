from abc import ABC

from strands.tools.decorator import DecoratedFunctionTool
from strands.types.tools import AgentTool


class ToolProvider(ABC):  # noqa: B024
    """Base class for creating stateful tool providers."""

    def get_tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, DecoratedFunctionTool):
                tools.append(attr)

        return tools
