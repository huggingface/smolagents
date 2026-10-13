# Inspecting runs with OpenTelemetry

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

## Why log your agent runs?

Agent runs are complicated to debug.

Validating that a run went properly is hard, since agent workflows are [unpredictable by design](../conceptual_guides/intro_agents) (if they were predictable, you'd just be using good old code). 

And inspecting a run is hard as well: multi-step agents tend to quickly fill a console with logs, and most of the errors are just "LLM dumb" kind of errors, from which the LLM auto-corrects in the next step by writing better code or tool calls.

So using instrumentation to record agent runs is necessary in production for later inspection and monitoring!

We've adopted the [OpenTelemetry](https://opentelemetry.io/) standard for instrumenting agent runs.

This means that you can just run some instrumentation code, then run your agents normally, and everything gets logged into your platform. Below are some examples of how to do this with different OpenTelemetry backends.

Here's how it then looks like on the platform:

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/inspect_run_phoenix.gif"/>
</div>


## Setting up telemetry with Arize AI Phoenix
First install the required packages. Here we install [Phoenix by Arize AI](https://github.com/Arize-ai/phoenix) because that's a good solution to collect and inspect the logs, but there are other OpenTelemetry-compatible platforms that you could use for this collection & inspection part.

```shell
pip install 'smolagents[telemetry,toolkit]'
```

Then run the collector in the background.

```shell
python -m phoenix.server.main serve
```

Finally, set up `SmolagentsInstrumentor` to trace your agents and send the traces to Phoenix default endpoint.

```python
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

register()
SmolagentsInstrumentor().instrument()
```
Then you can run your agents!

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    InferenceClientModel,
)

model = InferenceClientModel()

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run(
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
)
```
Voilà!
You can then navigate to `http://0.0.0.0:6006/projects/` to inspect your run!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/inspect_run_phoenix.png">

You can see that the CodeAgent called its managed ToolCallingAgent (by the way, the managed agent could have been a CodeAgent as well) to ask it to run the web search for the U.S. 2024 growth rate. Then the managed agent returned its report and the manager agent acted upon it to calculate the economy doubling time! Sweet, isn't it?

## Setting up telemetry with MLflow

MLflow has one-line autologging for Smolagents: it tracks runs, spans, inputs/outputs, and token usage in the MLflow UI.

Install MLflow, enable autologging, then run your agent with a couple of tools:

```python
%pip install mlflow smolagents

import mlflow
from smolagents import CodeAgent, ToolCallingAgent, WebSearchTool, VisitWebpageTool, InferenceClientModel

mlflow.smolagents.autolog()  # start tracing everything below

model = InferenceClientModel()
browser = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="Web search helper",
)
manager = CodeAgent(model=model, managed_agents=[browser])
manager.run("Find the latest US GDP growth rate and estimate when it would double.")
```

Start the UI to inspect traces, then open the Traces view in your browser:

```shell
mlflow server --port 5000
```

## Setting up telemetry with Confident AI

[Confident AI](https://www.confident-ai.com/) is an LLM observability and evaluation platform. Its OpenTelemetry-native tracing SDK, [`confident-trace`](https://github.com/confident-ai/confident-trace), detects smolagents on its own, so a single `init()` call traces agent runs along with the model and tool calls beneath them. You do not have to register an instrumentor or configure a tracer provider.

Install the SDK alongside smolagents and the provider SDK your model uses:

```shell
pip install confident-trace 'smolagents[openai]'
```

Set your [Confident AI project API key](https://app.confident-ai.com) and your provider key. Set `CONFIDENT_OTEL_ENDPOINT` to `https://eu.otel.confident-ai.com/v1/traces` if your project is in the EU region, otherwise traces are sent to the US servers:

```shell
export CONFIDENT_API_KEY="<your-confident-api-key>"
export OPENAI_API_KEY="<your-openai-key>"
```

Call `init()` once before you build an agent, and call `shutdown()` when the process exits so queued traces are flushed:

```python
import os

from confident_trace import init, shutdown
from smolagents import OpenAIModel, ToolCallingAgent

init()

agent = ToolCallingAgent(
    model=OpenAIModel("gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
    tools=[],
)

try:
    result = agent.run("Explain OpenTelemetry in one sentence.")
    print(result)
finally:
    shutdown()
```

smolagents spans describe the structure of the run. Model calls are captured by the provider integration underneath, so keep the integration for your provider SDK enabled to see LLM spans with messages, model details, and token usage. Traces appear in the Confident AI Observatory.

### Online evals

You can configure what happens to your incoming traces on Confident AI's [workflows page](https://www.confident-ai.com/docs/llm-tracing/workflows).

![Confident AI workflows page](https://confident-docs.s3.us-east-1.amazonaws.com/confident-trace-workflows.png)

- **Evaluation rules** — evaluate incoming traces against a [metric collection](https://www.confident-ai.com/docs/metrics/metric-collections).
- **Classifiers** — label your traces by issue, sentiment, or any dimension you define, which lets you group or filter them later on.
- **Queue ingestion** — route your production traces into annotation queues for your internal review team to annotate them by hand.
- **Dataset ingestion** — ingest your production traces into datasets to reuse as test cases and iterate on results to improve over time.

See the [Confident AI smolagents guide](https://www.confident-ai.com/docs/integrations/third-party/smolagents) for trace properties, multi-turn threads, and the rest of the configuration options.

## Setting up telemetry with 🪢 Langfuse

This part shows how to monitor and debug your Hugging Face **smolagents** with **Langfuse** using the `SmolagentsInstrumentor`.

> **What is Langfuse?** [Langfuse](https://langfuse.com) is an open-source platform for LLM engineering. It provides tracing and monitoring capabilities for AI agents, helping developers debug, analyze, and optimize their products. Langfuse integrates with various tools and frameworks via native integrations, OpenTelemetry, and SDKs.

### Step 1: Install Dependencies

```python
%pip install langfuse 'smolagents[telemetry]' openinference-instrumentation-smolagents
```

### Step 2: Set Up Environment Variables

Set your Langfuse API keys and configure the OpenTelemetry endpoint to send traces to Langfuse. Get your Langfuse API keys by signing up for [Langfuse Cloud](https://cloud.langfuse.com) or [self-hosting Langfuse](https://langfuse.com/self-hosting).

Also, add your [Hugging Face token](https://huggingface.co/settings/tokens) (`HF_TOKEN`) as an environment variable.

```python
import os
# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
 
# your Hugging Face token
os.environ["HF_TOKEN"] = "hf_..."
```

With the environment variables set, we can now initialize the Langfuse client. `get_client()` initializes the Langfuse client using the credentials provided in the environment variables.

```python
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
```

### Step 3: Initialize the `SmolagentsInstrumentor`

Initialize the `SmolagentsInstrumentor` before your application code. 


```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
 
SmolagentsInstrumentor().instrument()
```

### Step 4: Run your smolagent

```python
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    InferenceClientModel,
)

model = InferenceClientModel(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run(
    "How can Langfuse be used to monitor and improve the reasoning and decision-making of smolagents when they execute multi-step tasks, like dynamically adjusting a recipe based on user feedback or available ingredients?"
)
```

### Step 5: View Traces in Langfuse

After running the agent, you can view the traces generated by your smolagents application in [Langfuse](https://cloud.langfuse.com). You should see detailed steps of the LLM interactions, which can help you debug and optimize your AI agent.

![smolagents example trace](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

_[Public example trace in Langfuse](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/ce5160f9bfd5a6cd63b07d2bfcec6f54?timestamp=2025-02-11T09%3A25%3A45.163Z&display=details)_
