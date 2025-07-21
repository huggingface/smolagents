# Text-to-SQL

[[open-in-colab]]

이 튜토리얼에서는 `smolagents`를 사용해 SQL을 다루는 에이전트를 구현해보겠습니다.

> 먼저 중요한 질문으로 시작해봅시다: 그냥 간단하게 일반적인 text-to-SQL 파이프라인을 쓰면 안 될까요?

표준 text-to-sql 파이프라인은 안정성이 떨어집니다. 잘못된 쿼리가 생성될수도 있고, 더 나쁜 경우에는 그 쿼리가 오류를 일으키지 않고 잘못되거나 쓸모없는 결과를 반환할 수도 있습니다.

👉 반면, 에이전트 시스템은 출력 결과를 비판적으로 점검할 수 있고 쿼리를 수정할 필요가 있는지 스스로 결정할 수 있이 성능이 크게 향상됩니다.

이제 이 에이전트를 직접 만들어봅시다! 💪

아래 명령어를 실행해 필요한 의존성을 설치하세요:
```bash
!pip install smolagents python-dotenv sqlalchemy --upgrade -q
```

추론 프로바이더를 호출하려면 환경 변수 `HF_TOKEN`에 유효한 토큰이 설정돼있어야 합니다.
python-dotenv를 이용해 환경 변수를 불러오겠습니다.
```py
from dotenv import load_dotenv
load_dotenv()
```

다음으로, SQL 환경을 구성하겠습니다:
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

### 에이전트 구축하기

이제 툴을 활용해 SQL 테이블을 조회할 수 있도록 만들어봅시다.

툴의 description 속성은 에이전트 시스템에 의해 LLM 프롬프트에 포함되는 부분으로, LLM이 해당 도구를 어떻게 사용할 수 있는지에 대한 정보를 제공합니다. 바로 이 부분에 우리가 정의한 SQL 테이블의 설명을 작성하면 됩니다.

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
