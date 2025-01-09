import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import (
    CodeAgent,
    HfApiModel,
    tool,
    Tool,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    GradioUI
)

agent = CodeAgent(
    tools=[], model=HfApiModel(), max_steps=4, verbose=True
)

GradioUI(agent, UPLOAD_FOLDER='./data').launch()
