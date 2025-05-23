# How do multi-step agents work?

The ReAct framework ([Yao et al., 2022](https://huggingface.co/papers/2210.03629)) is currently the main approach to building agents.

The name is based on the concatenation of two words, "Reason" and "Act." Indeed, agents following this architecture will solve their task in as many steps as needed, each step consisting of a Reasoning step, then an Action step where it formulates tool calls that will bring it closer to solving the task at hand.

All agents in `smolagents` are based on singular `MultiStepAgent` class, which is an abstraction of ReAct framework.

On a basic level, this class performs actions on a cycle of following steps, where existing variables and knowledge is incorporated into the agent logs like below: 

Initialization: the system prompt is stored in a `SystemPromptStep`, and the user query is logged into a `TaskStep` .

While loop (ReAct loop):

- Use `agent.write_memory_to_messages()` to write the agent logs into a list of LLM-readable [chat messages](https://huggingface.co/docs/transformers/en/chat_templating).
- Send these messages to a `Model` object to get its completion. Parse the completion to get the action (a JSON blob for `ToolCallingAgent`, a code snippet for `CodeAgent`).
- Execute the action and logs result into memory (an `ActionStep`).
- At the end of each step, we run all callback functions defined in `agent.step_callbacks` .

Optionally, when planning is activated, a plan can be periodically revised and stored in a `PlanningStep` . This includes feeding facts about the task at hand to the memory.

For a `CodeAgent`, it looks like the figure below.

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png"
    />
</div>

Here is a video overview of how that works:

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

We implement two versions of agents:
- [`CodeAgent`] generates its tool calls as Python code snippets.
- [`ToolCallingAgent`] writes its tool calls as JSON, as is common in many frameworks. Depending on your needs, either approach can be used. For instance, web browsing often requires waiting after each page interaction, so JSON tool calls can fit well.

> [!TIP]
> Read [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) blog post to learn more about multi-step agents.
