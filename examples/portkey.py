from dotenv import load_dotenv
from smolagents import CodeAgent
from smolagents.models import PortkeyModel
# Load environment variables
load_dotenv()

model = PortkeyModel(
    model_id="gpt-4o",
    # API key can be set via PORTKEY_API_KEY env var
    # Virtual key can be set via PORTKEY_VIRTUAL_KEY_OPENAI env var or with other providers
)
  
# Create agent 
agent = CodeAgent(
    tools=[],
    model=model,
)

# Run agent - Osmosis integration happens automatically
result = agent.run("Write a function to calculate fibonacci numbers")