# Text-to-SQL

[[open-in-colab]]

इस ट्यूटोरियल में, हम देखेंगे कि कैसे `smolagents` का उपयोग करके एक एजेंट को SQL का उपयोग करने के लिए लागू किया जा सकता है।

> आइए सबसे महत्वपूर्ण प्रश्न से शुरू करें: इसे साधारण क्यों नहीं रखें और एक सामान्य text-to-SQL पाइपलाइन का उपयोग करें?

एक सामान्य text-to-SQL पाइपलाइन कमजोर होती है, क्योंकि उत्पन्न SQL क्वेरी गलत हो सकती है। इससे भी बुरी बात यह है कि क्वेरी गलत हो सकती है, लेकिन कोई एरर नहीं दिखाएगी, बल्कि बिना किसी अलार्म के गलत/बेकार आउटपुट दे सकती है।


👉 इसके बजाय, एक एजेंट सिस्टम आउटपुट का गंभीरता से निरीक्षण कर सकता है और तय कर सकता है कि क्वेरी को बदलने की जरूरत है या नहीं, इस प्रकार इसे बेहतर प्रदर्शन में मदद मिलती है।

आइए इस एजेंट को बनाएं! 💪

पहले, हम SQL एनवायरनमेंट सेटअप करते हैं:
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

### Agent बनाएं

अब आइए हमारी SQL टेबल को एक टूल द्वारा पुनर्प्राप्त करने योग्य बनाएं। 

टूल का विवरण विशेषता एजेंट सिस्टम द्वारा LLM के prompt में एम्बेड किया जाएगा: यह LLM को टूल का उपयोग करने के बारे में जानकारी देता है। यहीं पर हम SQL टेबल का वर्णन करना चाहते हैं।

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

अब आइए हमारा टूल बनाएं। इसे निम्नलिखित की आवश्यकता है: (अधिक जानकारी के लिए [टूल doc](../tutorials/tools) पढ़ें)
- एक डॉकस्ट्रिंग जिसमें आर्ग्युमेंट्स की सूची वाला `Args:` भाग हो।
- इनपुट और आउटपुट दोनों पर टाइप हिंट्स।

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

अब आइए एक एजेंट बनाएं जो इस टूल का लाभ उठाता है।

हम `CodeAgent` का उपयोग करते हैं, जो smolagents का मुख्य एजेंट क्लास है: एक एजेंट जो कोड में एक्शन लिखता है और ReAct फ्रेमवर्क के अनुसार पिछले आउटपुट पर पुनरावृत्ति कर सकता है।

मॉडल वह LLM है जो एजेंट सिस्टम को संचालित करता है। `InferenceClientModel` आपको HF के Inference API का उपयोग करके LLM को कॉल करने की अनुमति देता है, या तो सर्वरलेस या डेडिकेटेड एंडपॉइंट के माध्यम से, लेकिन आप किसी भी प्रोप्राइटरी API का भी उपयोग कर सकते हैं।

```py
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
```

### लेवल 2: टेबल जॉइन्स

अब आइए इसे और चुनौतीपूर्ण बनाएं! हम चाहते हैं कि हमारा एजेंट कई टेबल्स के बीच जॉइन को संभाल सके। 

तो आइए हम प्रत्येक receipt_id के लिए वेटर्स के नाम रिकॉर्ड करने वाली एक दूसरी टेबल बनाते हैं!

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
चूंकि हमने टेबल को बदल दिया है, हम LLM को इस टेबल की जानकारी का उचित उपयोग करने देने के लिए इस टेबल के विवरण के साथ `SQLExecutorTool` को अपडेट करते हैं।

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
चूंकि यह रिक्वेस्ट पिछले वाले से थोड़ी कठिन है, हम LLM इंजन को अधिक शक्तिशाली [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) का उपयोग करने के लिए स्विच करेंगे!

```py
sql_engine.description = updated_description

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="Qwen/Qwen3-4B-Instruct-2507"),
)

agent.run("Which waiter got more total money from tips?")
```
यह सीधे काम करता है! सेटअप आश्चर्यजनक रूप से सरल था, है ना?

यह उदाहरण पूरा हो गया! हमने इन अवधारणाओं को छुआ है:
- नए टूल्स का निर्माण।
- टूल के विवरण को अपडेट करना।
- एक मजबूत LLM में स्विच करने से एजेंट की तर्कशक्ति में मदद मिलती है।

✅ अब आप वह text-to-SQL सिस्टम बना सकते हैं जिसका आपने हमेशा सपना देखा है! ✨