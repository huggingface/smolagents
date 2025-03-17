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

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cloudpickle

from smolagents.agents import CodeAgent, ActionStep
from smolagents.utils import AgentExecutionError, AgentStoppedException, AgentPausedException


class PersistentCodeAgent(CodeAgent):
    """
    An extension of CodeAgent that supports state persistence through pause and resume functionality.
    
    This agent can handle two special exceptions:
    - AgentStoppedException: Gracefully stops execution
    - AgentPausedException: Pauses execution and saves the agent state for later resumption
    
    Args:
        storage_path (`str`, *optional*): Path where agent state will be saved when paused.
            Defaults to a temporary file in the current directory.
        **kwargs: All arguments accepted by CodeAgent.
    """
    
    def __init__(
        self,
        tools: List[Any],
        model: Callable[[List[Dict[str, str]]], Any],
        storage_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(tools=tools, model=model, **kwargs)
        self.storage_path = storage_path or os.path.join(os.getcwd(), "agent_state.pkl")
    
    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework with support for pause/stop exceptions.
        
        This method extends the CodeAgent step method to handle AgentStoppedException
        and AgentPausedException, which can be raised from callbacks.
        """
        try:
            return super().step(memory_step)
        except AgentStoppedException as e:
            # Log the stop event and gracefully terminate
            self.logger.log(f"Agent execution stopped: {e.message}", level=0)
            return f"Agent execution stopped: {e.message}"
        except AgentPausedException as e:
            # Save the agent state for later resumption
            self.logger.log(f"Agent execution paused: {e.message}", level=0)
            self._save_state(e.otel_context)
            return f"Agent execution paused: {e.message}"
    
    def _save_state(self, otel_context=None):
        """
        Save the agent's state to the storage path using cloudpickle.
        
        Args:
            otel_context: Optional OpenTelemetry context to save with the agent state.
        """
        state_data = {
            "agent": self,
            "otel_context": otel_context,
            "timestamp": time.time(),
        }
        
        try:
            with open(self.storage_path, "wb") as f:
                cloudpickle.dump(state_data, f)
            self.logger.log(f"Agent state saved to {self.storage_path}", level=0)
        except Exception as e:
            error_msg = f"Failed to save agent state: {str(e)}"
            self.logger.log(error_msg, level=0)
            raise AgentExecutionError(error_msg, self.logger)
    
    @classmethod
    def resume_execution(cls, storage_path: str, otel_context_handler=None):
        """
        Resume execution of a previously paused agent.
        
        Args:
            storage_path (`str`): Path to the saved agent state file.
            otel_context_handler (`Callable`, *optional*): Function to handle the OpenTelemetry context.
                If provided, it will be called with the saved OTEL context.
        
        Returns:
            PersistentCodeAgent: The restored agent instance.
        
        Raises:
            FileNotFoundError: If the storage path does not exist.
            AgentExecutionError: If there's an error loading the agent state.
        """
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Agent state file not found at {storage_path}")
        
        try:
            with open(storage_path, "rb") as f:
                state_data = cloudpickle.load(f)
            
            agent = state_data["agent"]
            otel_context = state_data["otel_context"]
            
            # Restore OTEL context if a handler is provided
            if otel_context_handler and otel_context:
                otel_context_handler(otel_context)
            
            agent.logger.log(f"Agent state restored from {storage_path}", level=0)
            return agent
        except Exception as e:
            raise AgentExecutionError(f"Failed to restore agent state: {str(e)}", agent.logger if 'agent' in locals() else None)