import logging
from unittest import TestCase
from unittest.mock import MagicMock, patch

import docker
from PIL import Image

from smolagents.remote_executors import DockerExecutor, E2BExecutor

from .utils.markers import require_run_all


class TestE2BExecutor:
    def test_e2b_executor_instantiation(self):
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            mock_sandbox.return_value.commands.run.return_value.error = None
            mock_sandbox.return_value.run_code.return_value.error = None
            executor = E2BExecutor(additional_imports=[], logger=logger)
        assert isinstance(executor, E2BExecutor)
        assert executor.logger == logger
        assert executor.final_answer_pattern.pattern == r"^final_answer\((.*)\)$"
        assert executor.sandbox == mock_sandbox.return_value


class TestDockerExecutor(TestCase):
    def setUp(self):
        self.logger = logging.getLogger("DockerExecutorTest")
        self.executor = DockerExecutor(
            additional_imports=["pillow", "numpy"], tools=[], logger=self.logger, initial_state={}
        )

    @require_run_all
    def test_initialization(self):
        """Check if DockerExecutor initializes without errors"""
        self.assertIsNotNone(self.executor.container, "Container should be initialized")

    @require_run_all
    def test_state_persistence(self):
        """Test that variables and imports form one snippet persist in the next"""
        code_action = "import numpy as np; a = 2"
        self.executor(code_action)

        code_action = "print(np.sqrt(a))"
        result, logs, final_answer = self.executor(code_action)
        assert "1.41421" in logs

    @require_run_all
    def test_execute_image_output(self):
        """Test execution that returns a base64 image"""
        code_action = """
import base64
from PIL import Image
from io import BytesIO

image = Image.new("RGB", (10, 10), (255, 0, 0))
final_answer(image)
"""
        result, logs, final_answer = self.executor(code_action)

        self.assertIsInstance(result, Image.Image, "Result should be a PIL Image")

    @require_run_all
    def test_syntax_error_handling(self):
        """Test handling of syntax errors"""
        code_action = 'print("Missing Parenthesis'  # Syntax error
        with self.assertRaises(ValueError) as context:
            self.executor(code_action)

        self.assertIn("SyntaxError", str(context.exception), "Should raise a syntax error")

    @require_run_all
    def test_cleanup_on_deletion(self):
        """Test if Docker container stops and removes on deletion"""
        container_id = self.executor.container.id
        self.executor.delete()  # Trigger cleanup

        client = docker.from_env()
        containers = [c.id for c in client.containers.list(all=True)]
        self.assertNotIn(container_id, containers, "Container should be removed")
