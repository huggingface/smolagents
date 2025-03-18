from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .memory import ActionStep, TaskStep
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from .utils import make_json_serializable
# Load environment variables
load_dotenv()

@dataclass
class OsmosisConfig:
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    base_url: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    agent_type: Optional[str] = "code_writing"

class OsmosisAPI:
    """Client for interacting with the Osmosis Agent Improvement API"""
    
    def __init__(self, config: Optional[OsmosisConfig] = None, api_key: Optional[str] = None, tenant_id: Optional[str] = None):
        """Initialize the Osmosis API client
        
        Args:
            config: Optional OsmosisConfig object containing API settings.
            api_key: Optional API key. If not provided, will look for OSMOSIS_API_KEY env var.
                     This is ignored if config is provided.
            tenant_id: Optional tenant ID. If not provided, will look for OSMOSIS_TENANT_ID env var.
                       This is ignored if config is provided.
        """
        if config is not None:
            self.config = config
        else:
            self.config = OsmosisConfig(api_key=api_key, tenant_id=tenant_id)
            
        self.api_key = self.config.api_key or os.getenv('OSMOSIS_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as OSMOSIS_API_KEY environment variable")
            
        self.tenant_id = self.config.tenant_id or os.getenv('OSMOSIS_TENANT_ID', self.api_key)
        self.base_url = self.config.base_url or "https://osmosis.gulp.dev"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

    def enhance_task(self, task: str, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Enhance a task with relevant knowledge
        
        Args:
            task: The task to enhance
            agent_type: Optional type of agent making request. Defaults to config.agent_type if not provided.
            
        Returns:
            Dict containing enhanced response and metadata
        """
        context = self.config.context or ""
        agent_type = agent_type or self.config.agent_type
        
        payload = {
            "tenant_id": self.tenant_id,
            "input_text": task + "\n\n" + str(context),
            "agent_type": agent_type
        }
        
        response = requests.post(
            f"{self.base_url}/enhance_task",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Return the entire response if 'response' key doesn't exist
        return result.get('response', result)

    def store_knowledge(self, query: str, turns: List[Dict[str, Any]], 
                       success: Optional[bool] = None,
                       agent_type: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store agent interactions and experiences
        
        Args:
            query: Original user query/instruction
            turns: Sequence of agent interaction turns
            success: Whether agent completed task successfully
            agent_type: Type of agent. Defaults to config.agent_type if not provided.
            metadata: Additional metadata about interaction
            
        Returns:
            Dict containing storage confirmation and metadata
        """
        # Ensure all data is JSON serializable
        serializable_turns = make_json_serializable(turns)
        agent_type = agent_type or self.config.agent_type
        
        # Create payload with serializable data
        payload = {
            "tenant_id": self.tenant_id,
            "query": query,
            "turns": serializable_turns,
            "success": success,
            "agent_type": agent_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": make_json_serializable(metadata or {})
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/store_knowledge",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except json.JSONDecodeError as e:
            print(f"JSON serialization error in store_knowledge: {e}")
            # Try again with string conversion as fallback
            payload["turns"] = [make_json_serializable(turn) for turn in turns]
            response = requests.post(
                f"{self.base_url}/store_knowledge",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    def delete_by_intent(self, intent: str, 
                        similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Delete knowledge entries matching an intent
        
        Args:
            intent: Intent to match against
            similarity_threshold: Similarity threshold for matching (0-1)
            
        Returns:
            Dict containing deletion results
        """
        params = {
            "tenant_id": self.tenant_id,
            "intent": intent,
            "similarity_threshold": similarity_threshold
        }
        
        response = requests.post(
            f"{self.base_url}/delete_by_intent",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def test_api_features(self):
        """Test all features of the Osmosis API"""
        try:
            # Test enhance_task
            print("\nTesting enhance_task...")
            enhance_response = self.enhance_task(
                task="Find and purchase a red Nike running shoe in size 10",
                agent_type="shopping_assistant"
            )
            print("Enhance task response:", enhance_response)

            # Test store_knowledge
            print("\nTesting store_knowledge...")
            turns = [
                {
                    "turn": 1,
                    "inputs": "On shopping website homepage. Need to search for red Nike running shoes.",
                    "decision": json.dumps({
                        "action_type": "click",
                        "element_name": "search_bar",
                        "text": ["Nike red running shoes"]
                    }),
                    "result": "Clicked search bar and entered search terms",
                    "memory": "User wants red Nike running shoes"
                }
            ]
            store_response = self.store_knowledge(
                query="Find red Nike running shoes",
                turns=turns,
                success=True,
                agent_type="shopping_assistant",
                metadata={"browser": "chrome", "session_id": "123"}
            )
            print("Store knowledge response:", store_response)

            # Test delete_by_intent
            print("\nTesting delete_by_intent...")
            delete_response = self.delete_by_intent(
                intent="find shoes",
                similarity_threshold=0.7
            )
            print("Delete by intent response:", delete_response)

            print("\nAll API tests completed successfully!")
            return True

        except Exception as e:
            print(f"Error during API testing: {str(e)}")
            return False

class OsmosisMixin:
    """Mixin class that adds Osmosis functionality to agent classes."""
    
    def __init__(self, osmosis_config: Optional[OsmosisConfig] = None, **kwargs):
        """Initialize the Osmosis mixin.
        
        Args:
            osmosis_config: Configuration for Osmosis support. If None, Osmosis functionality is disabled.
            **kwargs: Other keyword arguments to pass to the parent class.
        """
        # Initialize the parent class first
        super().__init__(**kwargs)
        
        # Setup Osmosis if config is provided
        self.osmosis = OsmosisAPI(config=osmosis_config) if osmosis_config is not None else None
    
    def enhance_task_with_osmosis(self, task: str) -> str:
        """Enhance a task with Osmosis knowledge if enabled.
        
        Args:
            task: The task to enhance
            
        Returns:
            Enhanced task or original task if Osmosis is disabled
        """
        if self.osmosis is not None:
            enhanced_task = self.osmosis.enhance_task(task=task)
            if enhanced_task:
                return enhanced_task
        return task
    
    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """Overrides the run method to use Osmosis functionality.
        
        This method enhances the task with Osmosis knowledge before running it,
        and stores the knowledge after completion.
        
        Args:
            task: The task to run
            stream: Whether to run in streaming mode
            reset: Whether to reset the agent's state
            images: Optional list of image paths to include
            additional_args: Optional additional arguments
            max_steps: Optional maximum number of steps to run
            
        Returns:
            The result of running the task
        """
        # Use the enhanced task if available
        enhanced_task = self.enhance_task_with_osmosis(task)
        
        # Get the normal result using the enhanced task
        result = super().run(
            task=enhanced_task,
            stream=stream,
            reset=reset,
            images=images,
            additional_args=additional_args,
            max_steps=max_steps
        )
        
        # Store knowledge if Osmosis is enabled
        if not stream:  # Only store knowledge for non-streaming runs
            self.store_knowledge_in_osmosis(task, result)
            
        return result
    
    def store_knowledge_in_osmosis(self, task: str, final_result: Any) -> None:
        """Store agent interactions and knowledge in Osmosis if enabled.
        
        Args:
            task: The original task/query
            final_result: The final result/answer from the agent
        """
        if self.osmosis is None:
            return
            
        # Convert memory steps to Osmosis turns format
        turns = self._get_turns_from_memory()
        
        # Make final_result JSON serializable
        serializable_result = make_json_serializable(final_result)
        
        # Add final result turn
        turns.append({
            "turn": len(turns) + 1,
            "inputs": "Final result",
            "decision": "Return final answer",
            "memory": "Task completed",
            "result": serializable_result
        })
        
        # Store knowledge in Osmosis
        self.osmosis.store_knowledge(
            query=task,
            turns=turns,
            success=True,  # We assume success since we got a final result
            agent_type=self.osmosis.config.agent_type
        )
    
    def _get_turns_from_memory(self) -> List[Dict]:
        """Convert agent memory steps to Osmosis turns format"""
        turns = []
        for step in self.memory.steps:
            if isinstance(step, TaskStep):
                turns.append({
                    "turn": len(turns) + 1,
                    "inputs": step.task,
                    "decision": "Task definition",
                    "memory": "Defined task",
                    "result": None
                })
            elif isinstance(step, ActionStep):
                turns.append({
                    "turn": len(turns) + 1,
                    "inputs": str(step.model_input_messages),
                    "decision": str(step.tool_calls) if step.tool_calls else step.model_output,
                    "memory": f"Step {step.step_number} execution",
                    "result": step.action_output
                })
        return turns

__all__ = [
    "OsmosisConfig",
    "OsmosisAPI",
    "OsmosisMixin"
]
