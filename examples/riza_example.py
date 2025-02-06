from smolagents import CodeAgent, HfApiModel
from smolagents.default_tools import VisitWebpageTool
from dotenv import load_dotenv

# Get an API key from https://dashboard.riza.io and put it in .env as RIZA_API_KEY=...

load_dotenv()

agent = CodeAgent(
    tools = [VisitWebpageTool()],
    model=HfApiModel(),
    additional_authorized_imports=["smolagents", "requests", "markdownify"],
    python_executor="riza",
)

agent.run(
    "How many appraisers are on this page https://www2.brea.ca.gov/breasearch/faces/party/search.xhtml"
)
