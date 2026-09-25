import sqlite3
from .tools import Tool

class SQLDatabaseTool(Tool):
    name = "sql_database"
    description = "Allows inspection and querying of a SQLite database in read-only mode. Use 'SCHEMA' to see tables."
    inputs = {
        "query": {"type": "string", "description": "The SQL query to run OR 'SCHEMA' to see tables."}
    }
    output_type = "string"

    def __init__(self, db_path: str, **kwargs):
        """
        Initialize the SQL tool.
        Args:
            db_path: Path to the SQLite database file (e.g., 'sales.db').
        """
        super().__init__(**kwargs)
        self.db_path = db_path

    def forward(self, query: str) -> str:
        """
        Executes a SQL query against the database and returns the results.
        If the query is 'SCHEMA', it returns the table definitions.
        Args:
            query: The SQL query to run OR 'SCHEMA' to see tables.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # God Mode Feature: Auto-Schema Inspection
            if query.strip().upper() == "SCHEMA":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                schema_info = []
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
                    schema_info.append(f"Table: {table_name}\nColumns: {', '.join(columns)}")
                conn.close()
                return "\n\n".join(schema_info)

            # Safety: Prevent modification (Read-Only Mode)
            if any(cmd in query.upper() for cmd in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER"]):
                conn.close()
                return "Error: This tool is currently in READ-ONLY mode for safety."

            # Run the actual query
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()

            # Format as a clean string for the LLM
            if not results:
                return "Query executed successfully but returned no results."
            
            # Simple CSV-style formatting
            output = [", ".join(columns)]
            for row in results:
                output.append(", ".join(map(str, row)))
            
            # Truncate if huge
            return "\n".join(output[:50]) # Limit to 50 rows to save context

        except Exception as e:
            return f"SQL Error: {str(e)}"
