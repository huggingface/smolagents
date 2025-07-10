import os

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer

# from langchain_openai import OpenAIEmbeddings
from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent


# from smolagents.agents import ToolCallingAgent


knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

## For your own PDFs, you can use the following code to load them into source_docs
# pdf_directory = "pdfs"
# pdf_files = [
#     os.path.join(pdf_directory, f)
#     for f in os.listdir(pdf_directory)
#     if f.endswith(".pdf")
# ]
# source_docs = []

# for file_path in pdf_files:
#     loader = PyPDFLoader(file_path)
#     docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)


print("Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)")
# Initialize embeddings and ChromaDB vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory="./chroma_db")


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


retriever_tool = RetrieverTool(vector_store)

# Choose which LLM engine to use!

# from smolagents import InferenceClientModel
# model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")

# from smolagents import TransformersModel
# model = TransformersModel(model_id="meta-llama/Llama-3.2-2B-Instruct")

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'
model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# # You can also use the ToolCallingAgent class
# agent = ToolCallingAgent(
#     tools=[retriever_tool],
#     model=model,
#     verbose=True,
# )

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
    stream_outputs=True,
)

agent_output = agent.run("How can I push a model to the Hub?")


print("Final output:")
print(agent_output)
