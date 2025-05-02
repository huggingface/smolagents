import io
from textwrap import dedent
from unittest.mock import MagicMock, patch

import docker
import PIL.Image
import pytest
from rich.console import Console

from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import DockerExecutor, E2BExecutor, RemotePythonExecutor, WebAssemblyExecutor
from smolagents.utils import AgentError

from .utils.markers import require_run_all


class TestRemotePythonExecutor:
    def test_send_tools_empty_tools(self, monkeypatch):
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_tools({})
        assert executor.run_code_raise_errors.call_count == 1
        # No new packages should be installed
        assert "!pip install" not in executor.run_code_raise_errors.call_args.args[0]


class TestE2BExecutorMock:
    def test_e2b_executor_instantiation(self):
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            mock_sandbox.return_value.commands.run.return_value.error = None
            mock_sandbox.return_value.run_code.return_value.error = None
            executor = E2BExecutor(
                additional_imports=[], logger=logger, api_key="dummy-api-key", template="dummy-template-id", timeout=60
            )
        assert isinstance(executor, E2BExecutor)
        assert executor.logger == logger
        assert executor.final_answer_pattern.pattern == r"^final_answer\((.*)\)$"
        assert executor.sandbox == mock_sandbox.return_value
        assert mock_sandbox.call_count == 1
        assert mock_sandbox.call_args.kwargs == {
            "api_key": "dummy-api-key",
            "template": "dummy-template-id",
            "timeout": 60,
        }


@pytest.fixture
def docker_executor():
    executor = DockerExecutor(
        additional_imports=["pillow", "numpy"],
        logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
    )
    yield executor
    executor.delete()


@require_run_all
class TestDockerExecutor:
    @pytest.fixture(autouse=True)
    def set_executor(self, docker_executor):
        self.executor = docker_executor

    def test_initialization(self):
        """Check if DockerExecutor initializes without errors"""
        assert self.executor.container is not None, "Container should be initialized"

    def test_state_persistence(self):
        """Test that variables and imports form one snippet persist in the next"""
        code_action = "import numpy as np; a = 2"
        self.executor(code_action)

        code_action = "print(np.sqrt(a))"
        result, logs, final_answer = self.executor(code_action)
        assert "1.41421" in logs

    def test_execute_output(self):
        """Test execution that returns a string"""
        code_action = 'final_answer("This is the final answer")'
        result, logs, final_answer = self.executor(code_action)
        assert result == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_multiline_output(self):
        """Test execution that returns a string"""
        code_action = 'result = "This is the final answer"\nfinal_answer(result)'
        result, logs, final_answer = self.executor(code_action)
        assert result == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_image_output(self):
        """Test execution that returns a base64 image"""
        code_action = dedent("""
            import base64
            from PIL import Image
            from io import BytesIO
            image = Image.new("RGB", (10, 10), (255, 0, 0))
            final_answer(image)
        """)
        result, logs, final_answer = self.executor(code_action)
        assert isinstance(result, PIL.Image.Image), "Result should be a PIL Image"

    def test_syntax_error_handling(self):
        """Test handling of syntax errors"""
        code_action = 'print("Missing Parenthesis'  # Syntax error
        with pytest.raises(AgentError) as exception_info:
            self.executor(code_action)
        assert "SyntaxError" in str(exception_info.value), "Should raise a syntax error"

    def test_cleanup_on_deletion(self):
        """Test if Docker container stops and removes on deletion"""
        container_id = self.executor.container.id
        self.executor.delete()  # Trigger cleanup

        client = docker.from_env()
        containers = [c.id for c in client.containers.list(all=True)]
        assert container_id not in containers, "Container should be removed"


class TestWebAssemblyExecutorUnit:
    def test_web_assembly_executor_instantiation(self):
        logger = MagicMock()

        # Mock subprocess.run to simulate Deno being installed
        with (
            patch("subprocess.run") as mock_run,
            patch("subprocess.Popen") as mock_popen,
            patch("requests.get") as mock_get,
            patch("time.sleep"),
        ):
            # Configure mocks
            mock_run.return_value.returncode = 0
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            mock_get.return_value.status_code = 200

            # Create the executor
            executor = WebAssemblyExecutor(additional_imports=["numpy", "pandas"], logger=logger, timeout=30)

            # Verify the executor was created correctly
            assert isinstance(executor, WebAssemblyExecutor)
            assert executor.logger == logger
            assert executor.timeout == 30
            assert "numpy" in executor.installed_packages
            assert "pandas" in executor.installed_packages

            # Verify Deno was checked
            assert mock_run.call_count == 1
            assert mock_run.call_args.args[0][0] == "deno"
            assert mock_run.call_args.args[0][1] == "--version"

            # Verify server was started
            assert mock_popen.call_count == 1
            assert mock_popen.call_args.args[0][0] == "deno"
            assert mock_popen.call_args.args[0][1] == "run"

            # Clean up
            with patch("shutil.rmtree"):
                executor.cleanup()


@require_run_all
class TestWebAssemblyExecutorIntegration:
    """
    Integration tests for WebAssemblyExecutor.

    These tests require Deno to be installed on the system.
    Skip these tests if you don't have Deno installed.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        try:
            # Check if Deno is installed
            import subprocess

            subprocess.run(["deno", "--version"], capture_output=True, check=True)

            # Create the executor
            self.executor = WebAssemblyExecutor(
                additional_imports=["numpy", "pandas"],
                logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
                timeout=60,
            )
            yield
            # Clean up
            self.executor.cleanup()
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.skip("Deno is not installed, skipping integration tests")

    def test_basic_execution(self):
        """Test basic code execution."""
        code = "a = 2 + 2; print(f'Result: {a}')"
        _, logs, _ = self.executor(code)
        assert "Result: 4" in logs

    def test_state_persistence(self):
        """Test that variables persist between executions."""
        # Define a variable
        self.executor("x = 42")

        # Use the variable in a subsequent execution
        _, logs, _ = self.executor("print(x)")
        assert "42" in logs

    def test_final_answer(self):
        """Test returning a final answer."""
        code = 'final_answer("This is the final answer")'
        result, _, is_final = self.executor(code)
        assert result == "This is the final answer"
        assert is_final is True

    def test_numpy_execution(self):
        """Test execution with NumPy."""
        code = """
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"Mean: {np.mean(arr)}")
        """
        _, logs, _ = self.executor(code)
        assert "Mean: 3.0" in logs

    def test_error_handling(self):
        """Test handling of Python errors."""
        code = "1/0"  # Division by zero
        with pytest.raises(AgentError) as excinfo:
            self.executor(code)
        assert "ZeroDivisionError" in str(excinfo.value)

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = "print('Missing parenthesis"  # Missing closing parenthesis
        with pytest.raises(AgentError) as excinfo:
            self.executor(code)
        assert "SyntaxError" in str(excinfo.value)
