<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Inspecting runs with OpenTelemetry

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

### Why log your agent runs?

Agent runs are complicated to debug.

Validating that a run went properly is hard, since agent workflows are [unpredictable by design](../conceptual_guides/intro_agents) (if they were predictable, you'd just be using good old code). 

And inspecting a run is hard as well: multi-step agents tend to quickly fill a console with logs, and most of the errors are just "LLM dumb" kind of errors, from which the LLM auto-corrects in the next step by writing better code or tool calls.

So using instrumentation to record agent runs is necessary in production for later inspection and monitoring!

We've adopted the [OpenTelemetry](https://opentelemetry.io/) standard for instrumenting agent runs.

This means that you can just run some instrumentation code, then run your agents normally, and everything gets logged into your platform.

Here's how it then looks like on the platform:

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/inspect_run_phoenix.gif"/>
</div>


### Setting up telemetry with Arize AI Phoenix
First install the required packages. Here we install [Phoenix by Arize AI](https://github.com/Arize-ai/phoenix) because that's a good solution to collect and inspect the logs, but there are other OpenTelemetry-compatible platforms that you could use for this collection & inspection part.

```shell
pip install smolagents
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

Then run the collector in the background.

```shell
python -m phoenix.server.main serve
```

Finally, set up `SmolagentsInstrumentor` to trace your agents and send the traces to Phoenix at the endpoint defined below.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://0.0.0.0:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```
Then you can run your agents!

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    ManagedAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel,
)

model = HfApiModel()

agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
)
managed_agent = ManagedAgent(
    agent=agent,
    name="managed_agent",
    description="This is an agent that can do web search.",
)
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_agent],
)
manager_agent.run(
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
)
```
Voilà!
You can then navigate to `http://0.0.0.0:6006/projects/` to inspect your run!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/inspect_run_phoenix.png">

You can see that the CodeAgent called its managed ToolCallingAgent (by the way, the managed agent could be have been a CodeAgent as well) to ask it to run the web search for the U.S. 2024 growth rate. Then the managed agent returned its report and the manager agent acted upon it to calculate the economy doubling time! Sweet, isn't it?