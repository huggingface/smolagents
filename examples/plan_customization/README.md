# Human-in-the-Loop: Customize Agent Plan Interactively

This example demonstrates advanced usage of the smolagents library, specifically showing how to implement Human-in-the-Loop strategies to:

1. **Interrupt agent execution after plan creation** using step callbacks
2. **Allow user interaction** to review and modify plans (Human-in-the-Loop)
3. **Resume execution** while preserving agent memory
4. **Modify plans in real-time** based on user feedback, keeping the human in control

## Human-in-the-Loop Key Features

### Interactive Plan Review
- The agent creates a plan and pauses execution
- Users can view the complete plan before execution begins
- Options to approve, modify, or cancel the plan

### Plan Modification
- Users can edit the agent's plan in real-time
- Modified plans are applied to the agent's memory
- Execution continues with the updated plan

### Memory Preservation
- Using `reset=False` preserves the agent's memory between runs
- Demonstrates how to build on previous interactions
- Shows memory state management across multiple executions
- Maintains transparency and control

## Usage

### Basic Usage
```python
python plan_customization.py
```

### Key Components

#### Step Callback Function
```python
def interrupt_after_plan(memory_step, agent):
    if isinstance(memory_step, PlanningStep):
        # Display plan and get user input
        # Modify plan if requested
        # Continue or interrupt based on user choice
```

#### Agent Configuration
```python
agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[DuckDuckGoSearchTool()],
    planning_interval=5,  # Plan every 5 steps
    step_callbacks={PlanningStep: interrupt_after_plan},  # Register callback for PlanningStep
    max_steps=10,
    verbosity_level=1
)
```

#### Resuming Execution
```python
# First run - may be interrupted
agent.run(task, reset=True)

# Resume with preserved memory
agent.run(task, reset=False)  # Keeps all previous steps
```

## Example Human-in-the-Loop Workflow

1. **Agent starts** with a complex task
2. **Planning step** is created automatically
3. **Execution pauses** for human review - step callback triggers
4. **Human-in-the-Loop**:
   1. **User reviews the plan** in a formatted display
   2. **User decides** to approve, modify, or cancel the plan
   3. **User modifies the plan** (if requested) - user can edit the plan
5. **Execution resumes** with approved/modified plan
6. **Memory preservation** - all steps are maintained for future runs, maintaining transparency and control

## Interactive Elements

### Plan Display
```
============================================================
🤖 AGENT PLAN CREATED
============================================================
1. Search for recent AI developments
2. Analyze the top results
3. Summarize the 3 most significant breakthroughs
4. Include sources for each breakthrough
============================================================
```

### User Choices
```
Choose an option:
1. Approve plan
2. Modify plan
3. Cancel
Your choice (1-3):
```

### Plan Modification Interface
```
----------------------------------------
MODIFY PLAN
----------------------------------------
Current plan: [displays current plan]
----------------------------------------
Enter your modified plan (press Enter twice to finish):
```

## Advanced Features

### Memory State Inspection
The example shows how to inspect the agent's memory:
```python
print(f"Current memory contains {len(agent.memory.steps)} steps:")
for i, step in enumerate(agent.memory.steps):
    step_type = type(step).__name__
    print(f"  {i+1}. {step_type}")
```

### Error Handling
Proper error handling for:
- User cancellation
- Plan modification errors
- Resume execution failures

## Requirements

- smolagents library
- DuckDuckGoSearchTool (included with smolagents)
- Access to InferenceClientModel (requires HuggingFace API token)

## Educational Value

This example teaches:
- **Step callback implementation** for custom agent behavior
- **Memory management** in multi-step agents
- **User interaction patterns** in agentic systems
- **Plan modification techniques** for dynamic agent control
- **Error handling** in interactive agent systems

Perfect for understanding how to build interactive, user-controlled AI agents that can adapt their behavior based on human feedback.
