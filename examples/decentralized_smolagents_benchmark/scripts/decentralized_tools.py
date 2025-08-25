from typing import Any

from smolagents.monitoring import LogLevel
from smolagents.tools import Tool


class FileReaderTool(Tool):
    """Tool for reading file contents."""

    name = "file_reader"
    inputs = {"file_path": {"type": "string", "description": "Path to the file to read"}}
    output_type = "string"
    description = "Read contents of a file"

    def forward(self, file_path: str) -> str:
        """Read contents of a file."""
        with open(file_path) as f:
            return f.read()


class SendMessageTool(Tool):
    """Tool for sending a message to another agent's queue."""

    name = "send_message"
    description = "Send a message to another agent via the shared queue_dict."  # TODO correct, sinon agents losts
    inputs = {
        "target_id": {"type": "integer", "description": "ID of the recipient agent"},
        "message": {"type": "any", "description": "Message to send"},
    }
    output_type = "null"

    def __init__(self, queue_dict: dict, agent_id: int, logger=None):
        """Initialize SendMessageTool.

        Args:
            queue_dict (dict): Dictionary mapping agent IDs to their message queues
            agent_id (int): ID of the sending agent
            logger (AgentLogger, optional): Logger for recording message activity
        """
        super().__init__()
        self.queue_dict = queue_dict
        self.agent_id = agent_id
        self.logger = logger

    def forward(self, target_id: int, message: Any) -> None:
        if target_id in self.queue_dict:
            self.queue_dict[target_id].put(message)
            if self.logger is not None:
                self.logger.log(
                    f"Agent {self.agent_id} sent message to Agent {target_id}",
                    level=LogLevel.INFO,
                )
        else:
            if self.logger is not None:
                self.logger.log(
                    f"Agent {self.agent_id} failed to send message: Target {target_id} not found",
                    level=LogLevel.WARNING,
                )
            raise ValueError(f"Target {target_id} not found in queue_dict")


class ReceiveMessagesTool(Tool):
    """Tool for receiving all messages from the current agent's queue."""

    name = "receive_messages"
    description = "Retrieve all messages for the current agent from its queue."
    inputs = {}
    output_type = "array"

    def __init__(self, queue_dict: dict, agent_id: int, logger=None):
        """Initialize ReceiveMessagesTool.

        Args:
            queue_dict (dict): Dictionary mapping agent IDs to their message queues
            agent_id (int): ID of the receiving agent
            logger (AgentLogger, optional): Logger for recording message activity
            process_message (Callable, optional): Function to process each received message
        """
        super().__init__()
        self.queue_dict = queue_dict
        self.agent_id = agent_id
        self.logger = logger
        self.queue = queue_dict[agent_id]

    def forward(self) -> list[Any]:
        messages = []
        while not self.queue.empty():
            msg = self.queue.get()
            if self.logger is not None:
                self.logger.log(f"Agent {self.agent_id} received message: {msg}", level=LogLevel.INFO)
            messages.append(msg)
        return messages
