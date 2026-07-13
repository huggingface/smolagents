import logging
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
import smolagents


LOGGER = logging.getLogger(__name__)


def configure_otlp(otlp_endpoint: str):
    LOGGER.info('Configuring OTLP: %r', otlp_endpoint)
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(otlp_endpoint)))
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def get_model() -> smolagents.Model:
    return smolagents.OpenAIModel(
        model_id='gpt-5-mini',
        api_key=os.environ['OPENAI_API_KEY']
    )


@smolagents.tool
def today_tool() -> str:
    """
    Provides the current date.
    """
    return "Today is 2026-01-29."


def create_agent() -> smolagents.CodeAgent:
    tools = [
        today_tool
    ]
    return smolagents.CodeAgent(
        tools=tools,
        max_steps=3,
        model=get_model()
    )


def main():
    configure_otlp('http://0.0.0.0:6006/v1/traces')
    agent = create_agent()
    agent.run("What is today's date?")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
