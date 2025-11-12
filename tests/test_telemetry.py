# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Source: https://github.com/Arize-ai/openinference/blob/main/python/instrumentation/openinference-instrumentation-smolagents/tests/openinference/instrumentation/smolagents/test_instrumentor.py

from typing import Generator

import pytest

from .utils.markers import require_run_all


# Add this at the module level to skip all tests if OpenTelemetry is not available
pytest.importorskip("opentelemetry", reason="requires opentelemetry")
pytest.importorskip(
    "openinference.instrumentation.smolagents", reason="requires openinference.instrumentation.smolagents"
)

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from smolagents.models import InferenceClientModel


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield
    SmolagentsInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@require_run_all
class TestOpenTelemetry:
    def test_model(self, in_memory_span_exporter: InMemorySpanExporter):
        model = InferenceClientModel()
        _ = model(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Who won the World Cup in 2018? Answer in one word with no punctuation.",
                        }
                    ],
                }
            ]
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "InferenceClientModel.generate"
        assert span.status.is_ok
        assert span.attributes
