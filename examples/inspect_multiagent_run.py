from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    ToolCallingAgent,
    VisitWebpageTool,
)


# Let's setup the instrumentation first

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter("http://0.0.0.0:6006/v1/traces")))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider, skip_dep_check=True)

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
