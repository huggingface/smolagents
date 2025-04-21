# Text-to-SQL

[[open-in-colab]]

在此教程中，我们将看到如何使用 `smolagents` 实现一个利用 SQL 的 agent。

> 让我们从经典问题开始：为什么不简单地使用标准的 text-to-SQL pipeline 呢？

标准的 text-to-SQL pipeline 很脆弱，因为生成的 SQL 查询可能会出错。更糟糕的是，查询可能出错却不引发错误警报，从而返回一些不正确或无用的结果。

👉 相反，agent 系统则可以检视输出结果并决定查询是否需要被更改，因此带来巨大的性能提升。

让我们来一起构建这个 agent! 💪

首先，我们构建一个 SQL 的环境：
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

# create city SQL table
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
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
```

### 构建 agent

现在，我们构建一个 agent，它将使用 SQL 查询来回答问题。工具的 description 属性将被 agent 系统嵌入到 LLM 的提示中：它为 LLM 提供有关如何使用该工具的信息。这正是我们描述 SQL 表的地方。

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

现在让我们构建我们的工具。它需要以下内容：（更多细节请参阅[工具文档](../tutorials/tools)）

- 一个带有 `Args:` 部分列出参数的 docstring。
- 输入和输出的type hints。

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

我们现在使用这个工具来创建一个 agent。我们使用 `CodeAgent`，这是 smolagent 的主要 agent 类：一个在代码中编写操作并根据 ReAct 框架迭代先前输出的 agent。

这个模型是驱动 agent 系统的 LLM。`InferenceClientModel` 允许你使用 HF  Inference API 调用 LLM，无论是通过 Serverless 还是 Dedicated endpoint，但你也可以使用任何专有 API。

```py
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
```

### Level 2: 表连接

现在让我们增加一些挑战！我们希望我们的 agent 能够处理跨多个表的连接。因此，我们创建一个新表，记录每个 receipt_id 的服务员名字！

```py
table_name = "waiters"
receipts = Table(
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
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
```

因为我们改变了表，我们需要更新 `SQLExecutorTool`，让 LLM 能够正确利用这个表的信息。

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

因为这个request 比之前的要难一些，我们将 LLM 引擎切换到更强大的 [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)！

```py
sql_engine.description = updated_description

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
)

agent.run("Which waiter got more total money from tips?")
```

它直接就能工作！设置过程非常简单，难道不是吗？

这个例子到此结束！我们涵盖了这些概念：

- 构建新工具。
- 更新工具的描述。
- 切换到更强大的 LLM 有助于 agent 推理。

✅ 现在你可以构建你一直梦寐以求的 text-to-SQL 系统了！✨
