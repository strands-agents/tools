import os

import pytest
from strands_tools.query_postgres import query_postgres  # update to actual import path


# Utility to simulate tool invocation
def run_tool(query, env=None, limit=None):
    # set tool_use_id arbitrarily
    tool_use_id = "test-invocation"
    if env:
        for k, v in env.items():
            os.environ[k] = v
    result = query_postgres(tool_use_id=tool_use_id, query=query, limit=limit or 100)
    return result


@pytest.fixture(autouse=True)
def clear_env():
    # clear environment variables for isolation
    for var in ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"):
        os.environ.pop(var, None)
    yield
    for var in ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"):
        os.environ.pop(var, None)


def test_missing_env_vars():
    res = run_tool("SELECT 1 as one;")
    assert res["toolUseId"] == "test-invocation"
    assert res["status"] == "error"
    assert "Could not connect" in res["content"][0]["text"]


def test_disallowed_query():
    # Provide env so connection is attempted but the query is blocked first
    env = {
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "testdb",
        "PGUSER": "user",
        "PGPASSWORD": "pwd",
    }
    res = run_tool("DELETE FROM users;", env=env)
    assert res["status"] == "error"
    assert "🚫 Only SELECT/CTE queries are allowed. This tool is read-only." in res["content"][0]["text"]


def test_read_only_select(monkeypatch):
    # --- Set env vars ---
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "testdb")
    monkeypatch.setenv("PGUSER", "readonly")
    monkeypatch.setenv("PGPASSWORD", "pwd")

    # --- Mock psycopg2 ---
    import sys
    import types

    # Fake cursor that simulates fetch
    class FakeCursor:
        def execute(self, q):
            self._rows = [{"col1": 123}, {"col1": 456}]
            self.description = [("col1",)]

        def fetchall(self):
            return self._rows

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Fake connection that returns fake cursor
    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

    # Create a fake psycopg2 module with connect + extras.RealDictCursor
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = lambda **kwargs: FakeConn()

    fake_extras = types.SimpleNamespace(RealDictCursor=object)
    fake_psycopg2.extras = fake_extras

    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras

    from strands_tools.query_postgres import query_postgres  # Import after patching

    result = query_postgres("test-invoke", "SELECT col1 FROM test")
    assert result["status"] == "success"
    assert (
        "\U0001f4ca Query: `SELECT col1 FROM test LIMIT 100`\n"
        "\U0001f9ee Rows: 2\n\U0001f520 Columns: col1\n \u2022 Row 1: col1=123\n \u2022 Row 2: col1=456"
        in result["content"][0]["text"]
    )
