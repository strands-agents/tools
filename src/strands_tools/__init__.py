"""Strands tools.

Tools are imported from their own modules (``from strands_tools import calculator``),
which binds the module rather than the tool inside it. Re-exporting the deprecated
tools under ``TYPE_CHECKING`` is what lets type checkers and IDEs resolve that name to
the ``@deprecated`` tool and flag the import. The block is typing-only, so it does not
run and the imported name stays a module at runtime.

Only names that resolve at runtime belong here. Re-exporting one that does not, such as
``slack_send_message``, would make a type checker accept an import that raises
ImportError.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_graph import agent_graph as agent_graph
    from .batch import batch as batch
    from .bright_data import bright_data as bright_data
    from .calculator import calculator as calculator
    from .chat_video import chat_video as chat_video
    from .cron import cron as cron
    from .current_time import current_time as current_time
    from .diagram import diagram as diagram
    from .editor import editor as editor
    from .elasticsearch_memory import elasticsearch_memory as elasticsearch_memory
    from .environment import environment as environment
    from .file_read import file_read as file_read
    from .file_write import file_write as file_write
    from .generate_image import generate_image as generate_image
    from .generate_image_stability import generate_image_stability as generate_image_stability
    from .graph import graph as graph
    from .handoff_to_user import handoff_to_user as handoff_to_user
    from .http_request import http_request as http_request
    from .image_reader import image_reader as image_reader
    from .journal import journal as journal
    from .load_tool import load_tool as load_tool
    from .mcp_client import mcp_client as mcp_client
    from .mem0_memory import mem0_memory as mem0_memory
    from .memory import memory as memory
    from .mongodb_memory import mongodb_memory as mongodb_memory
    from .nova_reels import nova_reels as nova_reels
    from .python_repl import python_repl as python_repl
    from .retrieve import retrieve as retrieve
    from .rss import rss as rss
    from .search_video import search_video as search_video
    from .shell import shell as shell
    from .slack import slack as slack
    from .sleep import sleep as sleep
    from .speak import speak as speak
    from .stop import stop as stop
    from .swarm import swarm as swarm
    from .think import think as think
    from .use_agent import use_agent as use_agent
    from .use_aws import use_aws as use_aws
    from .use_computer import use_computer as use_computer
    from .use_llm import use_llm as use_llm
    from .workflow import workflow as workflow
