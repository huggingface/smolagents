from dotenv import load_dotenv

from smolagents import CodeAgent, LiteLLMModel, GradioUI
from smolagents.default_tools import SGSmartScraperTool

load_dotenv()  # Load environment variables (SGAI_API_KEY is needed)

# Initialize the agent with both webpage tools
agent = CodeAgent(
    tools=[SGSmartScraperTool()],
    # model=HfApiModel("meta-llama/Llama-3.3-70B-Instruct"),  # You can use any model you prefer
    model=LiteLLMModel(model_id="gpt-4o-mini"),
    verbose=True,
)

result = agent.run(
    """
Visit https://huggingface.co/docs/transformers/index and extract:
1. The main title
2. The key features or main points
3. Any version information
Please present this information in a structured way.
"""
)
print("\nResult:", result)

# Try the agent in a Gradio UI
print("\nLaunching Gradio interface...")
GradioUI(agent).launch()
# try asking: what is https://scrapegraphai.com/ about?
