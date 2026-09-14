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
    from .batch import batch as batch
    from .bright_data import bright_data as bright_data
    from .calculator import calculator as calculator
    from .chat_video import chat_video as chat_video
    from .cron import cron as cron
    from .current_time import current_time as current_time
    from .diagram import diagram as diagram
    from .editor import editor as editor
    from .environment import environment as environment
    from .http_request import http_request as http_request
    from .exa import exa_get_contents as exa_get_contents
    from .exa import exa_search as exa_search
    from .journal import journal as journal
    from .memory import memory as memory
    from .retrieve import retrieve as retrieve
    from .rss import rss as rss
    from .search_video import search_video as search_video
    from .shell import shell as shell
    from .slack import slack as slack
    from .sleep import sleep as sleep
    from .tavily import tavily_crawl as tavily_crawl
    from .tavily import tavily_extract as tavily_extract
    from .tavily import tavily_map as tavily_map
    from .tavily import tavily_search as tavily_search
    from .think import think as think
