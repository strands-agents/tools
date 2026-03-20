"""
SQL Database Tool for Strands Agents.

Enables agents to connect to SQL databases (PostgreSQL, MySQL, SQLite),
run queries, and introspect schemas — with safe read-only mode by default.

Environment Variables:
    DATABASE_URL: Connection string (e.g. postgresql://user:pass@host/db)
                  Can also be passed directly via the `connection_string` param.

Usage with Strands Agent:
    from strands import Agent
    from strands_tools.sql_database import sql_database

    agent = Agent(tools=[sql_database])

    # List all tables
    agent.tool.sql_database(action="list_tables")

    # Describe a table's schema
    agent.tool.sql_database(action="describe_table", table="orders")

    # Run a SELECT query
    agent.tool.sql_database(
        action="query",
        sql="SELECT id, name FROM users LIMIT 5"
    )
"""

import logging
import os
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from strands import tool

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_engine(connection_string: str):
    """Return a SQLAlchemy engine; raises ImportError if sqlalchemy missing."""
    try:
        from sqlalchemy import create_engine, text  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy is required for the sql_database tool. Install it with: pip install 'strands-agents-tools[sql]'"
        ) from exc
    from sqlalchemy import create_engine

    return create_engine(connection_string, pool_pre_ping=True)


def _resolve_connection_string(connection_string: Optional[str]) -> str:
    """Resolve connection string from param or DATABASE_URL env var."""
    cs = connection_string or os.environ.get("DATABASE_URL", "")
    if not cs:
        raise ValueError(
            "No connection string provided. Pass `connection_string` or set the DATABASE_URL environment variable."
        )
    return cs


def _is_safe_query(sql: str) -> bool:
    """Return True if the SQL statement is read-only (SELECT / WITH / EXPLAIN)."""
    first_word = sql.strip().lstrip("(").split()[0].upper()
    return first_word in {"SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "DESC"}


def _rows_to_panel(rows, columns, title: str) -> Panel:
    """Render query results as a rich Panel containing a Table."""
    tbl = Table(show_header=True, header_style="bold cyan")
    for col in columns:
        tbl.add_column(str(col))
    for row in rows:
        tbl.add_row(*[str(v) if v is not None else "NULL" for v in row])
    return Panel(tbl, title=f"[bold cyan]{title}", border_style="cyan")


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------


def _list_tables(engine) -> dict[str, Any]:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    views = inspector.get_view_names()
    console.print(
        Panel(
            "\n".join(f"• {t}" for t in sorted(tables + views)) or "(none)",
            title="[bold cyan]Tables & Views",
            border_style="cyan",
        )
    )
    return {
        "status": "success",
        "tables": tables,
        "views": views,
        "content": [{"text": f"Found {len(tables)} table(s) and {len(views)} view(s)."}],
    }


def _describe_table(engine, table: str) -> dict[str, Any]:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    try:
        columns = inspector.get_columns(table)
        pk = inspector.get_pk_constraint(table)
        fks = inspector.get_foreign_keys(table)
    except Exception as exc:
        return {
            "status": "error",
            "content": [{"text": f"Could not describe table '{table}': {exc}"}],
        }

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Column")
    tbl.add_column("Type")
    tbl.add_column("Nullable")
    tbl.add_column("Default")
    tbl.add_column("PK")

    pk_cols = set(pk.get("constrained_columns", []))
    for col in columns:
        tbl.add_row(
            col["name"],
            str(col["type"]),
            "YES" if col.get("nullable", True) else "NO",
            str(col.get("default", "")) or "",
            "✓" if col["name"] in pk_cols else "",
        )

    console.print(Panel(tbl, title=f"[bold cyan]Schema: {table}", border_style="cyan"))

    fk_text = ""
    if fks:
        fk_lines = [f"  {fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}" for fk in fks]
        fk_text = "\nForeign Keys:\n" + "\n".join(fk_lines)
        console.print(fk_text)

    return {
        "status": "success",
        "table": table,
        "columns": [
            {
                "name": c["name"],
                "type": str(c["type"]),
                "nullable": c.get("nullable", True),
                "primary_key": c["name"] in pk_cols,
            }
            for c in columns
        ],
        "content": [{"text": f"Table '{table}' has {len(columns)} column(s).{fk_text}"}],
    }


def _run_query(engine, sql: str, read_only: bool, max_rows: int) -> dict[str, Any]:
    from sqlalchemy import text

    if read_only and not _is_safe_query(sql):
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "Blocked: only SELECT/WITH/EXPLAIN queries are allowed in "
                        "read-only mode. Set read_only=False to run write queries."
                    )
                }
            ],
        }

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        if result.returns_rows:
            columns = list(result.keys())
            rows = result.fetchmany(max_rows)
            console.print(_rows_to_panel(rows, columns, f"Results (up to {max_rows} rows)"))
            truncated = len(rows) == max_rows
            return {
                "status": "success",
                "columns": columns,
                "rows": [dict(zip(columns, row, strict=False)) for row in rows],
                "truncated": truncated,
                "content": [
                    {"text": (f"Returned {len(rows)} row(s)." + (" Results may be truncated." if truncated else ""))}
                ],
            }
        else:
            conn.commit()
            rowcount = result.rowcount
            console.print(
                Panel(
                    f"Query executed successfully. Rows affected: {rowcount}",
                    title="[bold green]Execute Result",
                    border_style="green",
                )
            )
            return {
                "status": "success",
                "rows_affected": rowcount,
                "content": [{"text": f"Query executed. Rows affected: {rowcount}"}],
            }


def _get_schema_summary(engine) -> dict[str, Any]:
    """Return a compact multi-table schema overview — great for Text-to-SQL context."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    summary: dict[str, list[str]] = {}
    lines = []
    for table in sorted(tables):
        cols = inspector.get_columns(table)
        col_strs = [f"{c['name']} {c['type']}" for c in cols]
        summary[table] = col_strs
        lines.append(f"{table}({', '.join(col_strs)})")

    console.print(
        Panel(
            "\n".join(lines) or "(no tables found)",
            title="[bold cyan]Schema Summary",
            border_style="cyan",
        )
    )
    return {
        "status": "success",
        "schema": summary,
        "content": [{"text": "\n".join(lines)}],
    }


# ---------------------------------------------------------------------------
# Main @tool entry point
# ---------------------------------------------------------------------------


@tool
def sql_database(
    action: str,
    connection_string: Optional[str] = None,
    sql: Optional[str] = None,
    table: Optional[str] = None,
    read_only: bool = True,
    max_rows: int = 100,
) -> dict[str, Any]:
    """
    Connect to a SQL database and perform queries or schema introspection.

    Supports PostgreSQL, MySQL, and SQLite via SQLAlchemy connection strings.
    Read-only mode is enabled by default to prevent accidental data modification.

    Args:
        action:            One of: "query", "execute", "list_tables",
                           "describe_table", "schema_summary"
        connection_string: SQLAlchemy connection string
                           (e.g. "sqlite:///mydb.db",
                                 "postgresql://user:pass@localhost/dbname",
                                 "mysql+pymysql://user:pass@localhost/dbname").
                           Falls back to DATABASE_URL env var if not provided.
        sql:               SQL statement to run (required for "query"/"execute").
        table:             Table name (required for "describe_table").
        read_only:         If True (default), only SELECT/WITH/EXPLAIN are allowed.
        max_rows:          Maximum number of rows to return for SELECT queries (default 100).

    Returns:
        dict with "status" ("success" | "error") and action-specific fields.

    Examples:
        # List all tables
        sql_database(action="list_tables")

        # Get full schema overview (ideal for Text-to-SQL context)
        sql_database(action="schema_summary")

        # Describe a single table
        sql_database(action="describe_table", table="orders")

        # Run a SELECT
        sql_database(action="query", sql="SELECT * FROM users LIMIT 10")

        # Run a write query (requires read_only=False)
        sql_database(
            action="execute",
            sql="UPDATE users SET active=1 WHERE id=42",
            read_only=False,
        )
    """
    try:
        cs = _resolve_connection_string(connection_string)
    except ValueError as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}

    try:
        engine = _get_engine(cs)
    except ImportError as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}
    except Exception as exc:
        return {
            "status": "error",
            "content": [{"text": f"Failed to create database engine: {exc}"}],
        }

    try:
        if action == "list_tables":
            return _list_tables(engine)

        elif action == "schema_summary":
            return _get_schema_summary(engine)

        elif action == "describe_table":
            if not table:
                return {
                    "status": "error",
                    "content": [{"text": "`table` parameter is required for describe_table."}],
                }
            return _describe_table(engine, table)

        elif action in ("query", "execute"):
            if not sql:
                return {
                    "status": "error",
                    "content": [{"text": "`sql` parameter is required for query/execute."}],
                }
            return _run_query(engine, sql.strip(), read_only, max_rows)

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action '{action}'. "
                            "Valid actions: list_tables, schema_summary, "
                            "describe_table, query, execute."
                        )
                    }
                ],
            }

    except Exception as exc:
        logger.exception("sql_database tool error")
        return {"status": "error", "content": [{"text": f"Unexpected error: {exc}"}]}
    finally:
        engine.dispose()
