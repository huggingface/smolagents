import sqlite3
import os
from smolagents.sql_tool import SQLDatabaseTool

def setup_dummy_db():
    """Creates a temporary database for testing."""
    conn = sqlite3.connect("test_data.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT, role TEXT)")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (1, 'Srijan', 'CEO')")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (2, 'Gemini', 'Assistant')")
    conn.commit()
    conn.close()

def test_sql_tool():
    setup_dummy_db()
    
    print("🔄 Initializing SQL Tool...")
    tool = SQLDatabaseTool(db_path="test_data.db")
    
    # Test 1: Get Schema
    print("🌍 Testing Schema Inspection...")
    schema = tool.forward("SCHEMA")
    print(f"Schema Result:\n{schema}\n")
    assert "users" in schema
    assert "role (TEXT)" in schema

    # Test 2: Run Query
    print("🌍 Testing Select Query...")
    result = tool.forward("SELECT * FROM users WHERE role='CEO'")
    print(f"Query Result:\n{result}\n")
    assert "Srijan" in result

    # Cleanup
    os.remove("test_data.db")
    print("✅ SQL Tool Verified.")

if __name__ == "__main__":
    test_sql_tool()
