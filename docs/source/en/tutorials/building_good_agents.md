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
# Building good agents

[[open-in-colab]]

There's a world of difference between building an agent that works and one that doesn't.
How can we build agents that fall into the latter category?
In this guide, we're going to talk about best practices for building agents.

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

### The best agentic systems are the simplest: simplify the workflow as much as you can

Giving an LLM some agency in your workflow introduces some risk of errors.

Well-programmed agentic systems have good error logging and retry mechanisms anyway, so the LLM engine has a chance to self-correct their mistake. But to reduce the risk of LLM error to the maximum, you should simplify your workflow!

Let's revisit the example from the [intro to agents](../conceptual_guides/intro_agents): a bot that answers user queries for a surf trip company.
Instead of letting the agent do 2 different calls for "travel distance API" and "weather API" each time they are asked about a new surf spot, you could just make one unified tool "return_spot_information", a function that calls both APIs at once and returns their concatenated outputs to the user.

This will reduce costs, latency, and error risk!

The main guideline is: Reduce the number of LLM calls as much as you can.

This leads to a few takeaways:
- Whenever possible, group 2 tools in one, like in our example of the two APIs.
- Whenever possible, logic should be based on deterministic functions rather than agentic decisions.

### Improve the information flow to the LLM engine

Remember that your LLM engine is like an *intelligent* robot, tapped into a room with the only communication with the outside world being notes passed under a door.

It won't know of anything that happened if you don't explicitly put that into its prompt.

So first start with making your task very clear!
Since an agent is powered by an LLM, minor variations in your task formulation might yield completely different results.

Then, improve the information flow towards your agent in tool use.

Particular guidelines to follow:
- Each tool should log (by simply using `print` statements inside the tool's `forward` method) everything that could be useful for the LLM engine.
  - In particular, logging detail on tool execution errors would help a lot!

For instance, here's a tool that retrieves weather data based on location and date-time:

First, here's a poor version:
```python
import datetime
from smolagents import tool

def get_weather_report_at_coordinates(coordinates, date_time):
    # Dummy function, returns a list of [temperature in °C, risk of rain on a scale 0-1, wave height in m]
    return [28.0, 0.35, 0.85]

def convert_location_to_coordinates(location):
    # Returns dummy coordinates
    return [3.3, -42.0]

@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for.
        date_time: the date and time for which you want the report.
    """
    lon, lat = convert_location_to_coordinates(location)
    date_time = datetime.strptime(date_time)
    return str(get_weather_report_at_coordinates((lon, lat), date_time))
```

Why is it bad?
- there's no precision of the format that should be used for `date_time`
- there's no detail on how location should be specified.
- there's no logging mechanism trying to make explicit failure cases like location not being in a proper format, or date_time not being properly formatted.
- the output format is hard to understand

If the tool call fails, the error trace logged in memory can help the LLM reverse engineer the tool to fix the errors. But why leave it with so much heavy lifting to do?

A better way to build this tool would have been the following:
```python
@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".
        date_time: the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    lon, lat = convert_location_to_coordinates(location)
    try:
        date_time = datetime.strptime(date_time)
    except Exception as e:
        raise ValueError("Conversion of `date_time` to datetime format failed, make sure to provide a string in format '%m/%d/%y %H:%M:%S'. Full trace:" + str(e))
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates((lon, lat), date_time)
    return f"Weather report for {location}, {date_time}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain*100:.0f}%, wave height is {wave_height}m."
```

In general, to ease the load on your LLM, the good question to ask yourself is: "How easy would it be for me, if I was dumb and using this tool for the first time ever, to program with this tool and correct my own errors?".

### Give more arguments to the agent

To pass some additional objects to your agent beyond the simple string describing the task, you can use the `additional_args` argument to pass any type of object:

```py
from smolagents import CodeAgent, HfApiModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

agent = CodeAgent(tools=[], model=HfApiModel(model_id=model_id), add_base_tools=True)

agent.run(
    "Why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
)
```
For instance, you can use this `additional_args` argument to pass images or strings that you want your agent to leverage.



## How to debug your agent

### 1. Use a stronger LLM

In an agentic workflows, some of the errors are actual errors, some other are the fault of your LLM engine not reasoning properly.
For instance, consider this trace for an `CodeAgent` that I asked to create a car picture:
```
==================================================================================================== New task ====================================================================================================
Make me a cool car picture
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
image_generator(prompt="A cool, futuristic sports car with LED headlights, aerodynamic design, and vibrant color, high-res, photorealistic")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Step 1:

- Time taken: 16.35 seconds
- Input tokens: 1,383
- Output tokens: 77
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
final_answer("/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Print outputs:

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Final answer:
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
```
The user sees, instead of an image being returned, a path being returned to them.
It could look like a bug from the system, but actually the agentic system didn't cause the error: it's just that the LLM brain did the mistake of not saving the image output into a variable.
Thus it cannot access the image again except by leveraging the path that was logged while saving the image, so it returns the path instead of an image.

The first step to debugging your agent is thus "Use a more powerful LLM". Alternatives like `Qwen2/5-72B-Instruct` wouldn't have made that mistake.

### 2. Provide more guidance / more information

You can also use less powerful models, provided you guide them more effectively.

Put yourself in the shoes of your model: if you were the model solving the task, would you struggle with the information available to you (from the system prompt + task formulation + tool description) ?

Would you need some added clarifications?

To provide extra information, we do not recommend to change the system prompt right away: the default system prompt has many adjustments that you do not want to mess up except if you understand the prompt very well.
Better ways to guide your LLM engine are:
- If it's about the task to solve: add all these details to the task. The task could be 100s of pages long.
- If it's about how to use tools: the description attribute of your tools.


### 3. Change the system prompt (generally not advised)

If above clarifications are not sufficient, you can change the system prompt.

Let's see how it works. For example, let us check the default system prompt for the [`CodeAgent`] (below version is shortened by skipping zero-shot examples).

```python
print(agent.system_prompt_template)
```
Here is what you get:
```text
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
{examples}

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:

{{tool_descriptions}}

{{managed_agents_descriptions}}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
```

As you can see, there are placeholders like `"{{tool_descriptions}}"`: these will be used upon agent initialization to insert certain automatically generated descriptions of tools or managed agents.

So while you can overwrite this system prompt template by passing your custom prompt as an argument to the `system_prompt` parameter, your new system prompt must contain the following placeholders:
- `"{{tool_descriptions}}"` to insert tool descriptions.
- `"{{managed_agents_description}}"` to insert the description for managed agents if there are any.
- For `CodeAgent` only: `"{{authorized_imports}}"` to insert the list of authorized imports.

Then you can change the system prompt as follows:

```py
from smolagents.prompts import CODE_SYSTEM_PROMPT

modified_system_prompt = CODE_SYSTEM_PROMPT + "\nHere you go!" # Change the system prompt here

agent = CodeAgent(
    tools=[], 
    model=HfApiModel(), 
    system_prompt=modified_system_prompt
)
```

This also works with the [`ToolCallingAgent`].


### 4. Extra planning

We provide a model for a supplementary planning step, that an agent can run regularly in-between normal action steps. In this step, there is no tool call, the LLM is simply asked to update a list of facts it knows and to reflect on what steps it should take next based on those facts.

```py
from smolagents import load_tool, CodeAgent, HfApiModel, DuckDuckGoSearchTool
from dotenv import load_dotenv

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = DuckDuckGoSearchTool()

agent = CodeAgent(
    tools=[search_tool],
    model=HfApiModel("Qwen/Qwen2.5-72B-Instruct"),
    planning_interval=3 # This is where you activate planning!
)

# Run it!
result = agent.run(
    "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
)
```
