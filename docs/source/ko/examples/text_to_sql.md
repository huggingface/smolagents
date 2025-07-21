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

### 에이전트 만들기

이제 툴을 활용해 SQL 테이블을 조회할 수 있도록 만들어봅시다.

툴의 설명 속성은 에이전트 시스템에 의해 LLM 프롬프트에 포함되는 부분으로, LLM이 해당 도구를 어떻게 사용할 수 있는지에 대한 정보를 제공합니다. 바로 이 부분에 우리가 정의한 SQL 테이블의 설명을 작성하면 됩니다.

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

이제 우리만의 툴을 만들어봅시다. 툴은 아래와 같은 요소를 필요로 합니다. (자세한 내용은 [툴 문서](../tutorials/tools)를 참고하세요)
- 인자(`Args:`) 목록이 포함된 독스트링
- 입력과 출력에 대한 타입 힌트

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

이제 이 툴을 활용하는 에이전트를 만들어보겠습니다.

여기서는 smolagent의 메인 에이전트 클래스인 `CodeAgent`를 사용합니다. `CodeAgent`는 코드로 액션을 작성하고 ReAct 프레임워크에 따라 이전 출력 결과를 반복적으로 개선할 수 있습니다.

모델은 에이전트 시스템을 구동하는 LLM을 의미합니다. `InferenceClientModel`을 사용하면 허깅페이스의 Inference API를 통해 서버리스 또는 Dedicated 엔드포인트 방식으로 LLM을 호출할 수 있으며, 필요에 따라 다른 사설 API를 사용할 수도 있습니다.

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
바로 동작합니다! 설정 과정이 놀라울 만큼 간단하지 않았나요?

이번 예제는 여기까지입니다! 지금까지 다음과 같은 개념들을 확인했습니다:
- 새로운 툴 만들기
- 툴 설명 업데이트하기
- 더 강력한 LLM을 사용할 경우 에이전트의 추론 능력이 향상된다는 점

✅ 이제 여러분이 꿈꿔왔던 text-to-SQL 시스템을 직접 만들어볼 수 있습니다! ✨
