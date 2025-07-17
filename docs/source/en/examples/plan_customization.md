# Human-in-the-Loop: Customize Agent Plan Interactively

This page demonstrates advanced usage of the smolagents library, with a special focus on **Human-in-the-Loop (HITL)** approaches for interactive plan creation, user-driven plan modification, and memory preservation in agentic workflows.
The example is based on the code in `examples/plan_customization/plan_customization.py`.

## Overview

This example teaches you how to implement Human-in-the-Loop strategies to:

- Interrupt agent execution after a plan is created (using step callbacks)
- Allow users to review and modify the agent's plan before execution (Human-in-the-Loop)
- Resume execution while preserving the agent's memory
- Dynamically update plans based on user feedback, keeping the human in control

## Key Concepts

### Step Callbacks for Plan Interruption

The agent is configured to pause after creating a plan. This is achieved by registering a step callback for the `PlanningStep`:

```python
agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[DuckDuckGoSearchTool()],
    planning_interval=5,  # Plan every 5 steps
    step_callbacks={PlanningStep: interrupt_after_plan},
    max_steps=10,
    verbosity_level=1
)
```

### Human-in-the-Loop: Interactive Plan Review and Modification

When the agent creates a plan, the callback displays it and prompts the human user to:

1. Approve the plan
2. Modify the plan
3. Cancel execution

Example interaction:

```
============================================================
🤖 AGENT PLAN CREATED
============================================================
1. Search for recent AI developments
2. Analyze the top results
3. Summarize the 3 most significant breakthroughs
4. Include sources for each breakthrough
============================================================

Choose an option:
1. Approve plan
2. Modify plan
3. Cancel
Your choice (1-3):
```

This Human-in-the-Loop step enables a human to intervene and review or modify the plan before execution continues, and ensures that the agent's actions align with human intent.

If the user chooses to modify, they can edit the plan directly. The updated plan is then used for subsequent execution steps.

### Memory Preservation and Resuming Execution

By running the agent with `reset=False`, all previous steps and memory are preserved. This allows you to resume execution after an interruption or plan modification:

```python
# First run (may be interrupted)
agent.run(task, reset=True)

# Resume with preserved memory
agent.run(task, reset=False)
```

### Inspecting Agent Memory

You can inspect the agent's memory to see all steps taken so far:

```python
print(f"Current memory contains {len(agent.memory.steps)} steps:")
for i, step in enumerate(agent.memory.steps):
    step_type = type(step).__name__
    print(f"  {i+1}. {step_type}")
```

## Example Human-in-the-Loop Workflow

1. Agent starts with a complex task
2. Planning step is created and execution pauses for human review
3. Human reviews and optionally modifies the plan (Human-in-the-Loop)
4. Execution resumes with the approved/modified plan
5. All steps are preserved for future runs, maintaining transparency and control

## Error Handling

The example includes error handling for:
- User cancellation
- Plan modification errors
- Resume execution failures

## Requirements

- smolagents library
- DuckDuckGoSearchTool (included with smolagents)
- InferenceClientModel (requires HuggingFace API token)

## Educational Value

This example demonstrates:
- Step callback implementation for custom agent behavior
- Memory management in multi-step agents
- User interaction patterns in agentic systems
- Plan modification techniques for dynamic agent control
- Error handling in interactive agent systems

---

For the full code, see [`examples/plan_customization`](https://github.com/huggingface/smolagents/tree/main/examples/plan_customization).
