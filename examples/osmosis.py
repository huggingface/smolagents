from dotenv import load_dotenv
from smolagents import CodeAgent, OsmosisConfig
from smolagents.models import LiteLLMModel
import os
# Load environment variables
load_dotenv()

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'
model = LiteLLMModel(
    model_id="openai/gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
)
  
def getCodeBase():
    return ""

# Configure Osmosis
osmosis_config = OsmosisConfig(
    api_key=os.environ.get("OSMOSIS_API_KEY"),  # You would need to set this env var
    tenant_id="tenant_id",  
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

# To run without Osmosis, simply don't provide the osmosis_config
agent_without_osmosis = CodeAgent(
    tools=[],
    model=model,
    # No osmosis_config means no Osmosis functionality
)