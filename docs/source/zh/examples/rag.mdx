# Agentic RAG

[[open-in-colab]]

Retrieval-Augmented-Generation (RAG) 是“使用大语言模型（LLM）来回答用户查询，但基于从知识库中检索的信息”。它比使用普通或微调的 LLM 具有许多优势：举几个例子，它允许将答案基于真实事实并减少虚构；它允许提供 LLM 领域特定的知识；并允许对知识库中的信息访问进行精细控制。

但是，普通的 RAG 存在一些局限性，以下两点尤为突出：

- 它只执行一次检索步骤：如果结果不好，生成的内容也会不好。
- 语义相似性是以用户查询为参考计算的，这可能不是最优的：例如，用户查询通常是一个问题，而包含真实答案的文档通常是肯定语态，因此其相似性得分会比其他以疑问形式呈现的源文档低，从而导致错失相关信息的风险。

我们可以通过制作一个 RAG  agent来缓解这些问题：非常简单，一个配备了检索工具的agent！这个 agent 将
会：✅ 自己构建查询和检索，✅ 如果需要的话会重新检索。

因此，它将比普通 RAG 更智能，因为它可以自己构建查询，而不是直接使用用户查询作为参考。这样，它可以更
接近目标文档，从而提高检索的准确性， [HyDE](https://huggingface.co/papers/2212.10496)。此 agent 可以
使用生成的片段，并在需要时重新检索，就像 [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/)。

我们现在开始构建这个系统. 🛠️

运行以下代码以安装所需的依赖包：
```bash
!pip install smolagents pandas langchain langchain-community sentence-transformers rank_bm25 --upgrade -q
```

你需要一个有效的 token 作为环境变量 `HF_TOKEN` 来调用 Inference Providers。我们使用 python-dotenv 来加载它。
```py
from dotenv import load_dotenv
load_dotenv()
```

我们首先加载一个知识库以在其上执行 RAG：此数据集是许多 Hugging Face 库的文档页面的汇编，存储为 markdown 格式。我们将仅保留 `transformers` 库的文档。然后通过处理数据集并将其存储到向量数据库中，为检索器准备知识库。我们将使用 [LangChain](https://python.langchain.com/docs/introduction/) 来利用其出色的向量数据库工具。
```py
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)
```

现在文档已准备好。我们来一起构建我们的 agent RAG 系统！
👉 我们只需要一个 RetrieverTool，我们的 agent 可以利用它从知识库中检索信息。

由于我们需要将 vectordb 添加为工具的属性，我们不能简单地使用带有 `@tool` 装饰器的简单工具构造函数：因此我们将遵循 [tools 教程](../tutorials/tools) 中突出显示的高级设置。

```py
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```
BM25 检索方法是一个经典的检索方法，因为它的设置速度非常快。为了提高检索准确性，你可以使用语义搜索，使用文档的向量表示替换 BM25：因此你可以前往 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 选择一个好的嵌入模型。

现在我们已经创建了一个可以从知识库中检索信息的工具，现在我们可以很容易地创建一个利用这个
`retriever_tool` 的 agent！此 agent 将使用如下参数初始化：
- `tools`：代理将能够调用的工具列表。
- `model`：为代理提供动力的 LLM。

我们的 `model` 必须是一个可调用对象，它接受一个消息的 list 作为输入，并返回文本。它还需要接受一个 stop_sequences 参数，指示何时停止生成。为了方便起见，我们直接使用包中提供的 `HfEngine` 类来获取调用 Hugging Face 的 Inference API 的 LLM 引擎。

接着，我们将使用 [meta-llama/Llama-3.3-70B-Instruct](meta-llama/Llama-3.3-70B-Instruct) 作为 llm 引
擎，因为：
- 它有一个长 128k 上下文，这对处理长源文档很有用。
- 它在 HF 的 Inference API 上始终免费提供！

_Note:_ 此 Inference API 托管基于各种标准的模型，部署的模型可能会在没有事先通知的情况下进行更新或替换。了解更多信息，请点击[这里](https://huggingface.co/docs/api-inference/supported-models)。

```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(
    tools=[retriever_tool], model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct"), max_steps=4, verbose=True
)
```

当我们初始化 CodeAgent 时，它已经自动获得了一个默认的系统提示，告诉 LLM 引擎按步骤处理并生成工具调用作为代码片段，但你可以根据需要替换此提示模板。接着，当其 `.run()` 方法被调用时，代理将负责调用 LLM 引擎，并在循环中执行工具调用，直到工具 `final_answer` 被调用，而其参数为最终答案。

```py
agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

print("Final output:")
print(agent_output)
```
