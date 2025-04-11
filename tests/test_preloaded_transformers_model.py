###############################
# ADD LOCAL PACKAGE TO SYSTEM #
###############################
import os
import sys

_actual_path    = os.path.dirname(os.path.abspath(__file__))
_path_local     = os.path.abspath(os.path.join(_actual_path, '..', 'src'))
sys.path.insert(0, _path_local)  

print(_path_local)

###############
# SETUP MODEL #
###############
import torch
from transformers   import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from smolagents     import CodeAgent, PreloadedTransformersModel

# --- Configuração do Qwen em 4-bit ---
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit                = True,
    bnb_4bit_quant_type         = "nf4",
    bnb_4bit_compute_dtype      = torch.bfloat16,
    bnb_4bit_use_double_quant   = True
)

tokenizer       = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_engine    = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True
)

model = PreloadedTransformersModel(
    model_engine,
    tokenizer = tokenizer
)

###########
# TESTE 1 #
###########
agent   = CodeAgent(tools=[], model=model)
result  = agent.run("Explain quantum mechanics in simple terms.")
print(result)

###########
# TESTE 2 #
###########
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