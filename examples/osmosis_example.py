from dotenv import load_dotenv
from smolagents import CodeAgent, OsmosisConfig
from smolagents.models import PortkeyModel

# Load environment variables
load_dotenv()

# Basic usage with Claude
model = PortkeyModel(
    model_id="gpt-4o",
    # API key can be set via PORTKEY_API_KEY env var
    # Virtual key can be set via PORTKEY_VIRTUAL_KEY_ANTHROPIC env var
)

def getCodeBase():
    return ""

# Configure Osmosis
osmosis_config = OsmosisConfig(
    enabled=True,
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