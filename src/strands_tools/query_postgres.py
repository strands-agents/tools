import os

from strands import tool


@tool
def query_postgres(tool_use_id: str, query: str, limit: int = 100) -> dict:
    """
    Safely execute **read-only** SQL queries (e.g., SELECT, WITH) against a PostgreSQL database.

    🔐 Security Guidelines:
    - This tool **strictly blocks** any non-read query such as INSERT, UPDATE, DELETE, DROP, ALTER, etc.
    - Use **read-only PostgreSQL credentials** (e.g., a user with SELECT-only permissions).
    - All connections should be made using **environment-controlled credentials** to avoid exposure in code.
    - Only use this tool for data exploration, reporting, and analytics — not transactional workloads.

    Parameters:
        tool_use_id: Unique ID for tool invocation (provided by the agent runtime)
        query: SQL SELECT/CTE query to execute
        limit: Optional row limit for SELECT queries (defaults to 100)

    Returns:
        A dict with toolUseId, status ("success" | "error"), and content (text response)
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "❌ psycopg2 not installed. Run: pip install psycopg2-binary"}],
        }

    # Sanitize query
    q_upper = query.strip().upper()
    disallowed = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "REPLACE", "GRANT", "REVOKE"]
    if any(q_upper.startswith(k) for k in disallowed):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "🚫 Only SELECT/CTE queries are allowed. This tool is read-only."}],
        }

    # Use env vars for connection
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            cursor_factory=RealDictCursor,
        )
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"❌ Could not connect to PostgreSQL: {str(e)}"}],
        }

    try:
        with conn:
            with conn.cursor() as cur:
                if q_upper.startswith("SELECT") and "LIMIT" not in q_upper:
                    query = f"{query.rstrip(';')} LIMIT {limit}"

                cur.execute(query)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description] if cur.description else []

                lines = [f"📊 Query: `{query}`", f"🧮 Rows: {len(rows)}", f"🔠 Columns: {', '.join(cols)}"]

                for i, row in enumerate(rows[: min(10, len(rows))], start=1):
                    lines.append(f" • Row {i}: " + ", ".join(f"{c}={row[c]}" for c in cols))
                if len(rows) > 10:
                    lines.append(f"...and {len(rows) - 10} more rows.")

                return {
                    "toolUseId": tool_use_id,
                    "status": "success",
                    "content": [{"text": "\n".join(lines)}],
                }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"❌ Query execution error: {str(e)}"}],
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass
