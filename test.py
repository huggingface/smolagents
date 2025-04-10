import sys
sys.path.insert(0, r"C:\Users\jonna\Desktop\projetos\2_HUGGING_FACE_AGENT_AI\2_1_THE_SMOLAGENS_FRAMEWORK\smolagents_v1\smolagents\src")  

from transformers   import BitsAndBytesConfig
from src.smolagents import CodeAgent, TransformersModel

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit                =True,
    bnb_4bit_compute_dtype      = "float16",
    bnb_4bit_use_double_quant   = True,
    bnb_4bit_quant_type         = "nf4"
)

model = TransformersModel(
    model_id,
    device_map          = "auto",
    torch_dtype         = "auto",  # pode ser omitido se quiser
    trust_remote_code   = True,
    model_config        = {'quantization_config': bnb_config},
    max_new_tokens      = 2000
)

'''
agent = CodeAgent(tools=[], model=model)
agent.run("Explain quantum mechanics in simple terms.")
'''

from langchain.docstore.document    import Document
from langchain.text_splitter        import RecursiveCharacterTextSplitter
from smolagents                     import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents                     import CodeAgent, LiteLLMModel, FinalAnswerTool

# tool
class PartyPlanningRetrieverTool(Tool):
    name        = "party_planning_retriever"
    description = "Uses semantic search to retrieve relevant party planning ideas for Alfred's superhero-themed party at Wayne Manor."
    inputs      = {
        "query": {
            "type":         "string",
            "description":  "The query to perform. This should be a query related to party planning or superhero themes."
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)

        return "\nRetrieved ideas:\n" + "".join([f"\n\n===== Idea {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)])


# doc
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in party_ideas]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size          = 500,
    chunk_overlap       = 50,
    add_start_index     = True,
    strip_whitespace    = True,
    separators          = ["\n\n", "\n", ".", " ", ""]
)
doc_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(doc_processed)

# Initialize the agned
agent = CodeAgent(tools=[party_planning_retriever, FinalAnswerTool()], model=model, verbosity_level = 3)

# Response
response = agent.run("Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options.")

print(response)
'''
{'Entertainment Ideas': ['Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.', 'Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.'], 'Catering Ideas': ["Serve dishes named after superheroes, such as 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'"], 'Decoration Ideas': ['A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.', 'Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.']}
'''