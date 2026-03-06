# 📚 Manage your agent's memory

[[open-in-colab]]

In the end, an agent can be defined by simple components: it has tools, prompts.
And most importantly, it has a memory of past steps, drawing a history of planning, execution, and errors.

### Replay your agent's memory

We propose several features to inspect a past agent run.

You can instrument the agent's run to display it in a great UI that lets you zoom in/out on specific steps, as highlighted in the [instrumentation guide](./inspect_runs).

You can also use `agent.replay()`, as follows:

After the agent has run:
```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=0)

result = agent.run("What's the 20th Fibonacci number?")
```

If you want to replay this last run, just use:
```py
agent.replay()
```

### Dynamically change the agent's memory

Many advanced use cases require dynamic modification of the agent's memory.

You can access the agent's memory using:

```py
from smolagents import ActionStep

system_prompt_step = agent.memory.system_prompt
print("The system prompt given to the agent was:")
print(system_prompt_step.system_prompt)

task_step = agent.memory.steps[0]
print("\n\nThe first task step was:")
print(task_step.task)

for step in agent.memory.steps:
    if isinstance(step, ActionStep):
        if step.error is not None:
            print(f"\nStep {step.step_number} got this error:\n{step.error}\n")
        else:
            print(f"\nStep {step.step_number} got these observations:\n{step.observations}\n")
```

Use `agent.memory.get_full_steps()` to get full steps as dictionaries.

You can also use step callbacks to dynamically change the agent's memory.

Step callbacks can access the `agent` itself in their arguments, so they can access any memory step as highlighted above, and change it if needed. For instance, let's say you are observing screenshots of each step performed by a web browser agent. You want to log the newest screenshot, and remove the images from ancient steps to save on token costs.

You could run something like the following.
_Note: this code is incomplete, some imports and object definitions have been removed for the sake of concision, visit [the original script](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py) to get the full working code._

```py
import helium
from PIL import Image
from io import BytesIO
from time import sleep

def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= latest_step - 2:
            previous_memory_step.observations_images = None
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]
```

Then you should pass this function in the `step_callbacks` argument upon initialization of your agent:

```py
CodeAgent(
    tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[update_screenshot],
    max_steps=20,
    verbosity_level=2,
)
```

Head to our [vision web browser code](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py) to see the full working example.

### Plug in external memory with lifecycle hooks

`AgentMemory` supports three optional lifecycle hooks that let you integrate external memory providers
(e.g., [mem0](https://github.com/mem0ai/mem0), Redis, SQLite, or file-based persistence)
without modifying core agent logic:

- **`on_run_start(task, memory)`** — Called at the start of each run, after the `TaskStep` is appended. Use this to load relevant context from a long-term memory store.
- **`on_step_added(step, memory)`** — Called after each step is appended. Use this to persist observations, tool outputs, or other step data.
- **`on_run_end(task, result, memory)`** — Called after the run completes. Use this to save run summaries, learned facts, or methodology outcomes.

All hooks are optional — when not provided, the agent behaves exactly as before.

#### Example: Cross-session memory with mem0

This example uses [mem0](https://github.com/mem0ai/mem0) to recall relevant context from previous runs
and persist new observations for future runs:

```python
from mem0 import Memory
from smolagents import CodeAgent, InferenceClientModel
from smolagents.memory import AgentMemory

mem = Memory()

def recall_context(task, memory):
    """Load relevant memories from previous runs into the current task context."""
    results = mem.search(task, user_id="agent")
    if results:
        context = "\n".join(m["memory"] for m in results)
        # Augment the task step with recalled context
        memory.steps[0].task += f"\n\nContext from previous runs:\n{context}"

def persist_observations(step, memory):
    """Save interesting observations to long-term memory after each step."""
    if hasattr(step, "observations") and step.observations:
        mem.add(step.observations[:500], user_id="agent")

def persist_result(task, result, memory):
    """Save a run summary to long-term memory after the run completes."""
    mem.add(f"Task: {task}\nResult: {result}", user_id="agent")

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    memory=AgentMemory(
        system_prompt="placeholder",  # Overwritten by the agent's own system prompt
        on_run_start=recall_context,
        on_step_added=persist_observations,
        on_run_end=persist_result,
    ),
)

# First run — no prior context
agent.run("Analyze sales data for Q1 trends")

# Second run — mem0 recalls Q1 insights automatically
agent.run("Now compare Q1 with Q2 trends")
```

#### Example: Filtered memory with LLM-as-a-judge

This example uses `on_step_added` to keep only the most relevant steps, scored by an LLM:

```python
from smolagents import CodeAgent, InferenceClientModel
from smolagents.memory import ActionStep, AgentMemory

model = InferenceClientModel()

def filter_irrelevant_steps(step, memory):
    """Remove steps with low relevance scores to keep memory lean."""
    if not isinstance(step, ActionStep) or not step.observations:
        return

    # Ask an LLM to score the relevance of this step's observations
    response = model.generate([{
        "role": "user",
        "content": (
            f"Rate the usefulness of this observation for solving the task on a scale of 1-10. "
            f"Reply with just the number.\n\n"
            f"Task: {memory.steps[0].task}\n"
            f"Observation: {step.observations[:500]}"
        ),
    }])

    try:
        score = int(response.content.strip())
    except (ValueError, AttributeError):
        return  # Keep the step if we can't parse the score

    if score < 4:
        # Replace low-value observations with a brief summary to save tokens
        step.observations = f"[Low relevance observation removed — score: {score}]"

agent = CodeAgent(
    tools=[...],
    model=model,
    memory=AgentMemory(
        system_prompt="placeholder",
        on_step_added=filter_irrelevant_steps,
    ),
)
```

#### Example: Simple JSON file persistence

For lightweight persistence without external dependencies:

```python
import json
from pathlib import Path
from smolagents import CodeAgent, InferenceClientModel
from smolagents.memory import ActionStep, AgentMemory

MEMORY_FILE = Path("agent_memory.json")

def save_result(task, result, memory):
    """Append each run's result to a JSON file for later review."""
    history = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else []
    history.append({"task": task, "result": str(result)})
    MEMORY_FILE.write_text(json.dumps(history, indent=2))

def load_history(task, memory):
    """Prepend past run results to the task context."""
    if MEMORY_FILE.exists():
        history = json.loads(MEMORY_FILE.read_text())
        if history:
            summary = "\n".join(f"- {h['task']}: {h['result']}" for h in history[-5:])
            memory.steps[0].task += f"\n\nPrevious run history:\n{summary}"

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    memory=AgentMemory(
        system_prompt="placeholder",
        on_run_start=load_history,
        on_run_end=save_result,
    ),
)
```

### Run agents one step at a time

This can be useful in case you have tool calls that take days: you can just run your agents step by step.
This will also let you update the memory on each step.

```py
from smolagents import InferenceClientModel, CodeAgent, ActionStep, TaskStep

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=1)
agent.python_executor.send_tools({**agent.tools})
print(agent.memory.system_prompt)

task = "What is the 20th Fibonacci number?"

# You could modify the memory as needed here by inputting the memory of another agent.
# agent.memory.steps = previous_agent.memory.steps

# Let's start a new task!
agent.memory.append_step(TaskStep(task=task, task_images=[]))

final_answer = None
step_number = 1
while final_answer is None and step_number <= 10:
    memory_step = ActionStep(
        step_number=step_number,
        observations_images=[],
    )
    # Run one step.
    final_answer = agent.step(memory_step)
    agent.memory.append_step(memory_step)
    step_number += 1

    # Change the memory as you please!
    # For instance to update the latest step:
    # agent.memory.steps[-1] = ...

print("The final answer is:", final_answer)
```
