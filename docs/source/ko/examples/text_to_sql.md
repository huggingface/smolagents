# Text-to-SQL

[[open-in-colab]]

In this tutorial, we’ll see how to implement an agent that leverages SQL using `smolagents`.

> Let's start with the golden question: why not keep it simple and use a standard text-to-SQL pipeline?

A standard text-to-sql pipeline is brittle, since the generated SQL query can be incorrect. Even worse, the query could be incorrect, but not raise an error, instead giving some incorrect/useless outputs without raising an alarm.

👉 Instead, an agent system is able to critically inspect outputs and decide if the query needs to be changed or not, thus giving it a huge performance boost.

Let’s build this agent! 💪

Run the line below to install required dependencies:
```bash
!pip install smolagents python-dotenv sqlalchemy --upgrade -q
```
To call Inference Providers, you will need a valid token as your environment variable `HF_TOKEN`.
We use python-dotenv to load it.
```py
from dotenv import load_dotenv
load_dotenv()
```

Then, we setup the SQL environment:
```py
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
    inspect,
    text,
)

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

def insert_rows_into_table(rows, table, engine=engine):
    for row in rows:
        stmt = insert(table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
]
insert_rows_into_table(rows, receipts)
```

### Build our agent

Now let’s make our SQL table retrievable by a tool.

The tool’s description attribute will be embedded in the LLM’s prompt by the agent system: it gives the LLM information about how to use the tool. This is where we want to describe the SQL table.

```py
inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]

table_description = "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
print(table_description)
```

```text
Columns:
  - receipt_id: INTEGER
  - customer_name: VARCHAR(16)
  - price: FLOAT
  - tip: FLOAT
```

Now let’s build our tool. It needs the following: (read [the tool doc](../tutorials/tools) for more detail)
- A docstring with an `Args:` part listing arguments.
- Type hints on both inputs and output.

```py
from smolagents import tool

@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output
```

Now let us create an agent that leverages this tool.

We use the `CodeAgent`, which is smolagents’ main agent class: an agent that writes actions in code and can iterate on previous output according to the ReAct framework.

The model is the LLM that powers the agent system. `InferenceClientModel` allows you to call LLMs using HF’s Inference API, either via Serverless or Dedicated endpoint, but you could also use any proprietary API.

```py
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.1-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
```

### Level 2: Table joins

Now let’s make it more challenging! We want our agent to handle joins across multiple tables.

So let’s make a second table recording the names of waiters for each receipt_id!

```py
table_name = "waiters"
waiters = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("waiter_name", String(16), primary_key=True),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "waiter_name": "Corey Johnson"},
    {"receipt_id": 2, "waiter_name": "Michael Watts"},
    {"receipt_id": 3, "waiter_name": "Michael Watts"},
    {"receipt_id": 4, "waiter_name": "Margaret James"},
]
insert_rows_into_table(rows, waiters)
```
Since we changed the table, we update the `SQLExecutorTool` with this table’s description to let the LLM properly leverage information from this table.

```py
updated_description = """Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
It can use the following tables:"""

inspector = inspect(engine)
for table in ["receipts", "waiters"]:
    columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table)]

    table_description = f"Table '{table}':\n"

    table_description += "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
    updated_description += "\n\n" + table_description

print(updated_description)
```
Since this request is a bit harder than the previous one, we’ll switch the LLM engine to use the more powerful [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)!

```py
sql_engine.description = updated_description

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
)

agent.run("Which waiter got more total money from tips?")
```
It directly works! The setup was surprisingly simple, wasn’t it?

This example is done! We've touched upon these concepts:
- Building new tools.
- Updating a tool's description.
- Switching to a stronger LLM helps agent reasoning.

✅ Now you can go build this text-to-SQL system you’ve always dreamt of! ✨
