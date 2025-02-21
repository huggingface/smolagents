from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register


register()
SmolagentsInstrumentor().instrument(skip_dep_check=True)


from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    ToolCallingAgent,
    VisitWebpageTool,
)


# Then we run the agentic part!
model = HfApiModel()

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run("If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?")
