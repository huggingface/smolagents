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
agent.memory.steps.append(TaskStep(task=task, task_images=[]))

final_answer = None
step_number = 1
while final_answer is None and step_number <= 10:
    memory_step = ActionStep(
        step_number=step_number,
        observations_images=[],
    )
    # Run one step.
    final_answer = agent.step(memory_step)
    agent.memory.steps.append(memory_step)
    step_number += 1

    # Change the memory as you please!
    # For instance to update the latest step:
    # agent.memory.steps[-1] = ...

print("The final answer is:", final_answer)
```
