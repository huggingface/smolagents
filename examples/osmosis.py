from dotenv import load_dotenv
from smolagents import CodeAgent, OsmosisConfig
from smolagents.models import LiteLLMModel
import os
# Load environment variables
load_dotenv()

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'
model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
)
  
def getCodeBase():
    return ""

# Configure Osmosis
osmosis_config = OsmosisConfig(
    enabled=True,
    tenant_id="tenant_id",  
    store_knowledge=True,
    enhance_tasks=True,
    context=str(getCodeBase()),
    agent_type="test"
)

# Create agent with Osmosis support
agent = CodeAgent(
    tools=[],
    model=model,
    osmosis_config=osmosis_config
)

# Run agent - Osmosis integration happens automatically
result = agent.run("Write a function to calculate fibonacci numbers")