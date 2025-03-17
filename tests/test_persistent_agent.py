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
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from smolagents import PersistentCodeAgent, Tool, ActionStep
from smolagents.utils import AgentPausedException, AgentStoppedException


class MockMessage:
    def __init__(self, content):
        self.content = content


def mock_model(messages):
    return MockMessage("""
```python
print("Test execution")
```
""")


class TestPersistentCodeAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "agent_state.pkl")
        
        # Create a simple agent for testing
        self.agent = PersistentCodeAgent(
            tools=[],
            model=mock_model,
            storage_path=self.storage_path,
        )
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly with storage path."""
        self.assertEqual(self.agent.storage_path, self.storage_path)
        self.assertIsInstance(self.agent, PersistentCodeAgent)
    
    def test_agent_stop_exception_handling(self):
        """Test that the agent handles AgentStoppedException correctly."""
        # Create a mock memory step
        memory_step = ActionStep(step_number=1)
        
        # Mock the super().step method to raise AgentStoppedException
        with patch('smolagents.agents.CodeAgent.step') as mock_super_step:
            mock_super_step.side_effect = AgentStoppedException("Test stop", self.agent.logger)
            
            # Call the step method
            result = self.agent.step(memory_step)
            
            # Verify the result contains the stop message
            self.assertIn("Agent execution stopped", result)
    
    def test_agent_pause_exception_handling(self):
        """Test that the agent handles AgentPausedException correctly."""
        # Create a mock memory step
        memory_step = ActionStep(step_number=1)
        
        # Mock the super().step method to raise AgentPausedException
        with patch('smolagents.agents.CodeAgent.step') as mock_super_step:
            mock_super_step.side_effect = AgentPausedException("Test pause", self.agent.logger)
            
            # Mock the _save_state method to avoid actual serialization
            with patch.object(self.agent, '_save_state') as mock_save:
                # Call the step method
                result = self.agent.step(memory_step)
                
                # Verify the result contains the pause message
                self.assertIn("Agent execution paused", result)
                
                # Verify _save_state was called
                mock_save.assert_called_once()
    
    def test_save_and_resume_state(self):
        """Test saving and resuming agent state."""
        # Mock cloudpickle.dump to avoid actual serialization
        with patch('cloudpickle.dump') as mock_dump:
            # Save the agent state
            self.agent._save_state({"test_context": "value"})
            
            # Verify cloudpickle.dump was called
            mock_dump.assert_called_once()
        
        # Mock open and cloudpickle.load for resume
        mock_state = {
            "agent": self.agent,
            "otel_context": {"test_context": "value"},
            "timestamp": 123456789,
        }
        
        with patch('builtins.open', MagicMock()):
            with patch('cloudpickle.load', return_value=mock_state):
                # Mock the otel_context_handler
                otel_handler = MagicMock()
                
                # Resume the agent
                with patch.object(self.agent.logger, 'log'):  # Mock logger.log
                    resumed_agent = PersistentCodeAgent.resume_execution(
                        self.storage_path, 
                        otel_context_handler=otel_handler
                    )
                
                # Verify the agent was restored
                self.assertIsInstance(resumed_agent, PersistentCodeAgent)
                
                # Verify the otel_context_handler was called with the context
                otel_handler.assert_called_once_with({"test_context": "value"})
    
    def test_resume_with_missing_file(self):
        """Test resuming with a missing state file."""
        non_existent_path = os.path.join(self.temp_dir.name, "non_existent.pkl")
        
        # Verify that attempting to resume from a non-existent file raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            PersistentCodeAgent.resume_execution(non_existent_path)


if __name__ == "__main__":
    unittest.main()