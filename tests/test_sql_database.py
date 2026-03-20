"""
Tests for the sql_database tool.

Uses pytest tmp_path fixture for isolated, parallel-safe SQLite databases.
Compatible with pytest-xdist parallel execution.
"""

import pytest
from sqlalchemy import create_engine, text

from strands_tools.sql_database import sql_database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_url(tmp_path):
    """Create a seeded SQLite DB in a unique temp directory per test worker."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"

    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE users (
                    id      INTEGER PRIMARY KEY,
                    name    TEXT    NOT NULL,
                    email   TEXT    UNIQUE,
                    active  INTEGER DEFAULT 1
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE orders (
                    id      INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    amount  REAL    NOT NULL
                )
                """
            )
        )
        conn.execute(text("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 1)"))
        conn.execute(text("INSERT INTO users VALUES (2, 'Bob',   'bob@example.com',   0)"))
        conn.execute(text("INSERT INTO orders VALUES (1, 1, 99.99)"))
        conn.execute(text("INSERT INTO orders VALUES (2, 2, 49.50)"))
        conn.commit()
    engine.dispose()

    return url


def run(db_url, action, **kwargs):
    return sql_database(action=action, connection_string=db_url, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListTables:
    def test_returns_success(self, db_url):
        result = run(db_url, "list_tables")
        assert result["status"] == "success"

    def test_contains_expected_tables(self, db_url):
        result = run(db_url, "list_tables")
        assert "users" in result["tables"]
        assert "orders" in result["tables"]


class TestDescribeTable:
    def test_describes_users(self, db_url):
        result = run(db_url, "describe_table", table="users")
        assert result["status"] == "success"
        col_names = [c["name"] for c in result["columns"]]
        assert "id" in col_names
        assert "email" in col_names

    def test_primary_key_flagged(self, db_url):
        result = run(db_url, "describe_table", table="users")
        id_col = next(c for c in result["columns"] if c["name"] == "id")
        assert id_col["primary_key"] is True

    def test_missing_table_param(self, db_url):
        result = run(db_url, "describe_table")
        assert result["status"] == "error"
        assert "table" in result["content"][0]["text"].lower()

    def test_nonexistent_table(self, db_url):
        result = run(db_url, "describe_table", table="nonexistent_xyz")
        assert result["status"] == "error"


class TestQuery:
    def test_select_all_users(self, db_url):
        result = run(db_url, "query", sql="SELECT * FROM users")
        assert result["status"] == "success"
        assert len(result["rows"]) == 2

    def test_columns_returned(self, db_url):
        result = run(db_url, "query", sql="SELECT id, name FROM users")
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    def test_max_rows_respected(self, db_url):
        result = run(db_url, "query", sql="SELECT * FROM users", max_rows=1)
        assert len(result["rows"]) == 1
        assert result["truncated"] is True

    def test_read_only_blocks_insert(self, db_url):
        result = run(
            db_url,
            "query",
            sql="INSERT INTO users VALUES (99, 'X', 'x@x.com', 1)",
            read_only=True,
        )
        assert result["status"] == "error"
        assert "read-only" in result["content"][0]["text"].lower()

    def test_empty_sql_rejected(self, db_url):
        result = run(db_url, "query", sql="")
        assert result["status"] == "error"

    def test_missing_sql_rejected(self, db_url):
        result = run(db_url, "query")
        assert result["status"] == "error"


class TestSchemaSummary:
    def test_returns_success(self, db_url):
        result = run(db_url, "schema_summary")
        assert result["status"] == "success"

    def test_all_tables_present(self, db_url):
        result = run(db_url, "schema_summary")
        assert "users" in result["schema"]
        assert "orders" in result["schema"]

    def test_columns_in_summary(self, db_url):
        result = run(db_url, "schema_summary")
        users_cols = result["schema"]["users"]
        assert any("name" in c for c in users_cols)


class TestConnectionHandling:
    def test_missing_connection_string_error(self):
        import os

        os.environ.pop("DATABASE_URL", None)
        result = sql_database(action="list_tables", connection_string=None)
        assert result["status"] == "error"
        assert "connection" in result["content"][0]["text"].lower()

    def test_invalid_connection_string_error(self):
        result = sql_database(
            action="list_tables",
            connection_string="notavaliddriver://??",
        )
        assert result["status"] == "error"


class TestUnknownAction:
    def test_unknown_action_returns_error(self, db_url):
        result = run(db_url, "fly_to_moon")
        assert result["status"] == "error"
        assert "unknown action" in result["content"][0]["text"].lower()
