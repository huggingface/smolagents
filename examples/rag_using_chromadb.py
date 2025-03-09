import os
import os.path as osp
import argparse

from dotenv import load_dotenv

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer

# from langchain_openai import OpenAIEmbeddings
from smolagents import LiteLLMModel, Tool, GradioUI
from smolagents.agents import CodeAgent
# from smolagents.agents import ToolCallingAgent


def prepare_docs():
    print('preparing docs ...(might take 5-20 min depending on your hardware\n')
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
    
    return docs_processed


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


def parse_args():
    help_msg = """\
Agentic RAG with ChromaDB
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args
    # For anthropic: change args below to 'LiteLLM', 'anthropic/claude-3-5-sonnet-20240620' and "ANTHROPIC_API_KEY"
    parser.add_argument('--model_src', type=str, default="LiteLLM", choices=["HfApi", "LiteLLM", "Transformers"])
    parser.add_argument('--model', type=str, default="groq/qwen-2.5-coder-32b")
    parser.add_argument('--LiteLLMModel_API_key_name', type=str, default="GROQ_API_KEY")
    parser.add_argument('--emb_func', type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2"]) # feel free to add support for more embedding functions
    parser.add_argument('--persist_dir', type=str, default="./chroma_db", help='Path to the persisted vector DB')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Initialize embedding function and ChromaDB vector store
    embedding_function = HuggingFaceEmbeddings(model_name=args.emb_func)

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if osp.exists(args.persist_dir):
        print('using persisted data')
        vector_store = Chroma(embedding_function=embedding_function,
                            persist_directory=args.persist_dir)
    else:
        docs_processed = prepare_docs()
        vector_store = Chroma.from_documents(documents=docs_processed,
                                            embedding=embedding_function,
                                            persist_directory=args.persist_dir)

    retriever_tool = RetrieverTool(vector_store)


    # Choose which LLM engine to use!
    if args.model_src == 'HfApi':
        from smolagents import HfApiModel
        # You can choose to not pass any model_id to HfApiModel to use a default free model
        model = HfApiModel(model_id=args.model,
                           token=os.environ.get('HF_API_KEY') )

    elif args.model_src == 'Transformers':
        from smolagents import TransformersModel
        model = TransformersModel(model_id=args.model)

    elif args.model_src == 'LiteLLM':
        model = LiteLLMModel(model_id=args.model,
                             api_key=os.environ.get(args.LiteLLMModel_API_key_name) )
    else:
        raise ValueError('Choose the models source from ["HfApi", "LiteLLM", "Transformers"]')

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
        verbosity_level=2
    )

    GradioUI(agent).launch()
    # try: How can I push a model to the Hub?


if __name__ == '__main__':
    load_dotenv()
    main()
