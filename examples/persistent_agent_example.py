#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Example demonstrating the PersistentCodeAgent with pause and resume functionality.
"""

import os
import time
from typing import Dict, List

from smolagents import PersistentCodeAgent, Tool, ActionStep
from smolagents.utils import AgentPausedException, AgentStoppedException

# Mock model for demonstration purposes
def mock_model(messages: List[Dict[str, str]]):
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    # Return a simple Python code that will trigger our pause/stop logic
    return MockMessage("""
```python
# This is a simple example to demonstrate pause/stop functionality
import time

print("Starting execution...")
time.sleep(1)

# This will trigger the pause callback after 3 steps
if step_number == 3:
    print("About to pause execution")
    raise_pause_exception()

# This will trigger the stop callback after 5 steps
if step_number == 5:
    print("About to stop execution")
    raise_stop_exception()

print(f"Completed step {step_number}")
```
""")

# Custom tools for pausing and stopping the agent
class PauseTool(Tool):
    name = "raise_pause_exception"
    description = "Pauses the agent execution for later resumption"
    
    def __call__(self, agent=None):
        if agent:
            # In a real implementation, you might want to capture OTEL context here
            otel_context = {"trace_id": "mock-trace-id", "span_id": "mock-span-id"}
            raise AgentPausedException("Agent paused by user request", agent.logger, None, otel_context)
        return "Agent would be paused if running in an agent context"

class StopTool(Tool):
    name = "raise_stop_exception"
    description = "Stops the agent execution"
    
    def __call__(self, agent=None):
        if agent:
            raise AgentStoppedException("Agent stopped by user request", agent.logger)
        return "Agent would be stopped if running in an agent context"

# Custom callback to track steps and demonstrate pause/resume
def step_callback(memory_step: ActionStep, agent=None):
    print(f"Step {memory_step.step_number} completed")
    # Access step_number in the agent's Python environment
    if agent and hasattr(agent, "python_executor"):
        agent.python_executor.send_variables({"step_number": memory_step.step_number})

def main():
    # Create storage path
    storage_path = os.path.join(os.getcwd(), "agent_state.pkl")
    
    # Create tools
    tools = [PauseTool(), StopTool()]
    
    # Create agent
    agent = PersistentCodeAgent(
        tools=tools,
        model=mock_model,
        storage_path=storage_path,
        step_callbacks=[step_callback],
        additional_authorized_imports=["time"],
    )
    
    try:
        # Run the agent - it will pause at step 3
        print("\n=== Starting agent execution ===")
        result = agent.run("Execute a task with pause and resume capability")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Agent execution interrupted: {str(e)}")
    
    # Check if the agent state was saved
    if os.path.exists(storage_path):
        print(f"\n=== Agent state saved to {storage_path} ===")
        
        # Wait a moment to simulate time passing
        time.sleep(2)
        
        # Resume the agent
        print("\n=== Resuming agent execution ===")
        resumed_agent = PersistentCodeAgent.resume_execution(storage_path)
        
        try:
            # Continue execution - it will stop at step 5
            result = resumed_agent.run(resumed_agent.task, reset=False)
            print(f"Final result: {result}")
        except Exception as e:
            print(f"Resumed agent execution interrupted: {str(e)}")

if __name__ == "__main__":
    main()