import importlib
import io
from textwrap import dedent
from unittest.mock import MagicMock, patch

import docker
import PIL.Image
import pytest
from rich.console import Console

from smolagents.default_tools import FinalAnswerTool, WikipediaSearchTool
from smolagents.local_python_executor import CodeOutput
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import (
    BlaxelExecutor,
    DockerExecutor,
    E2BExecutor,
    ModalExecutor,
    RemotePythonExecutor,
    WasmExecutor,
)
from smolagents.utils import AgentError

from .utils.markers import require_run_all


class TestRemotePythonExecutor:
    def test_send_tools_empty_tools(self):
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_tools({})
        assert executor.run_code_raise_errors.call_count == 1
        # No new packages should be installed
        assert "!pip install" not in executor.run_code_raise_errors.call_args.args[0]

    def test_send_variables_with_empty_dict_is_noop(self):
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_variables({})
        assert executor.run_code_raise_errors.call_count == 0

    @require_run_all
    def test_send_tools_with_default_wikipedia_search_tool(self):
        tool = WikipediaSearchTool()
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_tools({"wikipedia_search": tool})
        assert executor.run_code_raise_errors.call_count == 2
        assert "!pip install wikipedia-api" == executor.run_code_raise_errors.call_args_list[0].args[0]
        assert "class WikipediaSearchTool(Tool)" in executor.run_code_raise_errors.call_args_list[1].args[0]


class TestE2BExecutorUnit:
    def test_e2b_executor_instantiation(self):
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            mock_sandbox.return_value.commands.run.return_value.error = None
            mock_sandbox.return_value.run_code.return_value.error = None
            # Also set up v2 path in case Sandbox.create is used
            mock_sandbox.create.return_value.commands.run.return_value.error = None
            mock_sandbox.create.return_value.run_code.return_value.error = None
            executor = E2BExecutor(
                additional_imports=[], logger=logger, api_key="dummy-api-key", template="dummy-template-id", timeout=60
            )
        assert isinstance(executor, E2BExecutor)
        assert executor.logger == logger
        # Support both e2b v1 (Sandbox(...)) and v2 (Sandbox.create(...))
        if mock_sandbox.create.called:
            sandbox_obj = mock_sandbox.create.return_value
            called_ctor = mock_sandbox.create
        else:
            sandbox_obj = mock_sandbox.return_value
            called_ctor = mock_sandbox
        assert executor.sandbox == sandbox_obj
        assert called_ctor.call_count == 1
        assert called_ctor.call_args.kwargs == {
            "api_key": "dummy-api-key",
            "template": "dummy-template-id",
            "timeout": 60,
        }

    def test_cleanup(self):
        """Test that the cleanup method properly shuts down the sandbox"""
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            # Setup mock
            mock_sandbox.return_value.kill = MagicMock()
            # Also set up v2 path in case Sandbox.create is used
            mock_sandbox.create.return_value.kill = MagicMock()

            # Create executor
            executor = E2BExecutor(additional_imports=[], logger=logger, api_key="dummy-api-key")

            # Call cleanup
            executor.cleanup()

            # Verify sandbox was killed
            if mock_sandbox.create.called:
                mock_sandbox.create.return_value.kill.assert_called_once()
            else:
                mock_sandbox.return_value.kill.assert_called_once()
            assert logger.log.call_count >= 2  # Should log start and completion messages


@pytest.fixture
def e2b_executor():
    executor = E2BExecutor(
        additional_imports=["pillow", "numpy"],
        logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
    )
    yield executor
    executor.cleanup()


@require_run_all
class TestE2BExecutorIntegration:
    @pytest.fixture(autouse=True)
    def set_executor(self, e2b_executor):
        self.executor = e2b_executor

    @pytest.mark.parametrize(
        "code_action, expected_result",
        [
            (
                dedent('''
                    final_answer("""This is
                    a multiline
                    final answer""")
                '''),
                "This is\na multiline\nfinal answer",
            ),
            (
                dedent("""
                    text = '''Text containing
                    final_answer(5)
                    '''
                    final_answer(text)
                """),
                "Text containing\nfinal_answer(5)\n",
            ),
            (
                dedent("""
                    num = 2
                    if num == 1:
                        final_answer("One")
                    elif num == 2:
                        final_answer("Two")
                """),
                "Two",
            ),
        ],
    )
    def test_final_answer_patterns(self, code_action, expected_result):
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == expected_result

    def test_custom_final_answer(self):
        class CustomFinalAnswerTool(FinalAnswerTool):
            def forward(self, answer: str) -> str:
                return "CUSTOM" + answer

        self.executor.send_tools({"final_answer": CustomFinalAnswerTool()})
        code_action = dedent("""
            final_answer(answer="_answer")
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "CUSTOM_answer"

    def test_custom_final_answer_with_custom_inputs(self):
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                return answer1 + "CUSTOM" + answer2

        self.executor.send_tools({"final_answer": CustomFinalAnswerToolWithCustomInputs()})
        code_action = dedent("""
            final_answer(
                answer1="answer1_",
                answer2="_answer2"
            )
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "answer1_CUSTOM_answer2"


class TestDockerExecutorUnit:
    def test_cleanup(self):
        """Test that cleanup properly stops and removes the container"""
        logger = MagicMock()
        with (
            patch("docker.from_env") as mock_docker_client,
            patch("requests.post") as mock_post,
            patch("websocket.create_connection"),
        ):
            # Setup mocks
            mock_container = MagicMock()
            mock_container.status = "running"
            mock_container.short_id = "test123"

            mock_docker_client.return_value.containers.run.return_value = mock_container
            mock_docker_client.return_value.images.get.return_value = MagicMock()

            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {"id": "test-kernel-id"}

            # Create executor
            executor = DockerExecutor(additional_imports=[], logger=logger, build_new_image=False)

            # Call cleanup
            executor.cleanup()

            # Verify container was stopped and removed
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()


class CommonDockerExecutorIntegration:
    @pytest.fixture(autouse=True)
    def set_executor(self, custom_executor):
        self.executor = custom_executor

    def test_state_persistence(self):
        """Test that variables and imports form one snippet persist in the next"""
        code_action = "import numpy as np; a = 2"
        self.executor(code_action)

        code_action = "print(np.sqrt(a))"
        code_output = self.executor(code_action)
        assert "1.41421" in code_output.logs

    def test_execute_output(self):
        """Test execution that returns a string"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = 'final_answer("This is the final answer")'
        code_output = self.executor(code_action)
        assert code_output.output == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_multiline_output(self):
        """Test execution that returns a string"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = 'result = "This is the final answer"\nfinal_answer(result)'
        code_output = self.executor(code_action)
        assert code_output.output == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_image_output(self):
        """Test execution that returns a base64 image"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = dedent("""
            import base64
            from PIL import Image
            from io import BytesIO
            image = Image.new("RGB", (10, 10), (255, 0, 0))
            final_answer(image)
        """)
        code_output = self.executor(code_action)
        assert isinstance(code_output.output, PIL.Image.Image), "Result should be a PIL Image"

    def test_syntax_error_handling(self):
        """Test handling of syntax errors"""
        code_action = 'print("Missing Parenthesis'  # Syntax error
        with pytest.raises(AgentError) as exception_info:
            self.executor(code_action)
        assert "SyntaxError" in str(exception_info.value), "Should raise a syntax error"

    @pytest.mark.parametrize(
        "code_action, expected_result",
        [
            (
                dedent('''
                    final_answer("""This is
                    a multiline
                    final answer""")
                '''),
                "This is\na multiline\nfinal answer",
            ),
            (
                dedent("""
                    text = '''Text containing
                    final_answer(5)
                    '''
                    final_answer(text)
                """),
                "Text containing\nfinal_answer(5)\n",
            ),
            (
                dedent("""
                    num = 2
                    if num == 1:
                        final_answer("One")
                    elif num == 2:
                        final_answer("Two")
                """),
                "Two",
            ),
        ],
    )
    def test_final_answer_patterns(self, code_action, expected_result):
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == expected_result

    def test_custom_final_answer(self):
        class CustomFinalAnswerTool(FinalAnswerTool):
            def forward(self, answer: str) -> str:
                return "CUSTOM" + answer

        self.executor.send_tools({"final_answer": CustomFinalAnswerTool()})
        code_action = dedent("""
            final_answer(answer="_answer")
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "CUSTOM_answer"

    def test_custom_final_answer_with_custom_inputs(self):
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                return answer1 + "CUSTOM" + answer2

        self.executor.send_tools({"final_answer": CustomFinalAnswerToolWithCustomInputs()})
        code_action = dedent("""
            final_answer(
                answer1="answer1_",
                answer2="_answer2"
            )
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "answer1_CUSTOM_answer2"


@require_run_all
class TestDockerExecutorIntegration(CommonDockerExecutorIntegration):
    @pytest.fixture
    def custom_executor(self):
        executor = DockerExecutor(
            additional_imports=["pillow", "numpy"],
            logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
        )
        yield executor
        executor.delete()

    def test_initialization(self):
        """Check if DockerExecutor initializes without errors"""
        assert self.executor.container is not None, "Container should be initialized"

    def test_cleanup_on_deletion(self):
        """Test if Docker container stops and removes on deletion"""
        container_id = self.executor.container.id
        self.executor.delete()  # Trigger cleanup

        client = docker.from_env()
        containers = [c.id for c in client.containers.list(all=True)]
        assert container_id not in containers, "Container should be removed"


@require_run_all
class TestModalExecutorIntegration(CommonDockerExecutorIntegration):
    @pytest.fixture
    def custom_executor(self):
        executor = ModalExecutor(
            additional_imports=["pillow", "numpy"],
            logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
        )
        yield executor
        executor.delete()


class TestModalExecutorUnit:
    @patch("smolagents.remote_executors._websocket_run_code_raise_errors")
    @patch("requests.post")
    @patch("requests.get")
    @patch("websocket.create_connection")
    @patch("modal.App.lookup")
    @patch("modal.Sandbox.create")
    def test_sandbox_lifecycle(
        self, mock_sandbox_create, mock_app_lookup, mock_create_connection, mock_get, mock_post, mock_run_code_raises
    ):
        """Test that sandbox is created with the correct kwargs and cleaned up correctly."""
        modal = pytest.importorskip("modal")
        port = 8889

        logger = MagicMock()
        mock_sandbox = MagicMock()
        tunnel_mock = MagicMock()
        tunnel_mock.host = "r4234.modal.host"
        mock_sandbox.tunnels.return_value = {port: tunnel_mock}

        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": "test-kernel-id"}
        mock_run_code_raises.return_value = CodeOutput(output="3", logs="", is_final_answer=False)
        mock_sandbox_create.return_value = mock_sandbox

        executor = ModalExecutor(
            additional_imports=[],
            logger=logger,
            app_name="my-custom-app-name",
            port=port,
            create_kwargs={
                "secrets": [modal.Secret.from_dict({"MY_SECRET": "ABC"})],
                "timeout": 100,
                "cpu": 2,
            },
        )

        create_call = mock_sandbox_create.mock_calls[0]
        assert create_call.args == (
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip='0.0.0.0'",
            f"--KernelGatewayApp.port={port}",
            "--KernelGatewayApp.allow_origin='*'",
        )
        assert create_call.kwargs["timeout"] == 100
        assert create_call.kwargs["cpu"] == 2
        assert len(create_call.kwargs["secrets"]) == 2
        mock_app_lookup.assert_called_with("my-custom-app-name", create_if_missing=True)

        executor.run_code_raise_errors("1 + 2")
        executor.cleanup()
        mock_sandbox.terminate.assert_called()


class TestWasmExecutorUnit:
    def test_wasm_executor_instantiation(self):
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
            executor = WasmExecutor(additional_imports=["numpy", "pandas"], logger=logger, timeout=30)

            # Verify the executor was created correctly
            assert isinstance(executor, WasmExecutor)
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
class TestWasmExecutorIntegration:
    """
    Integration tests for WasmExecutor.

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
            self.executor = WasmExecutor(
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
        code_output = self.executor(code)
        assert "Result: 4" in code_output.logs

    def test_state_persistence(self):
        """Test that variables persist between executions."""
        # Define a variable
        self.executor("x = 42")

        # Use the variable in a subsequent execution
        code_output = self.executor("print(x)")
        assert "42" in code_output.logs

    def test_final_answer(self):
        """Test returning a final answer."""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code = 'final_answer("This is the final answer")'
        code_output = self.executor(code)
        assert code_output.output == "This is the final answer"
        assert code_output.is_final_answer is True

    def test_numpy_execution(self):
        """Test execution with NumPy."""
        code = """
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"Mean: {np.mean(arr)}")
        """
        code_output = self.executor(code)
        assert "Mean: 3.0" in code_output.logs

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


class TestBlaxelExecutorUnit:
    def test_blaxel_executor_instantiation_without_blaxel_sdk(self):
        """Test that BlaxelExecutor raises appropriate error when blaxel SDK is not installed."""
        logger = MagicMock()
        with patch.dict("sys.modules", {"blaxel.core": None}):
            with pytest.raises(ModuleNotFoundError) as excinfo:
                BlaxelExecutor(additional_imports=[], logger=logger)
            assert "Please install 'blaxel' extra" in str(excinfo.value)

    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_instantiation_with_blaxel_sdk(
        self, mock_settings, mock_sandbox_instance, mock_create_kernel
    ):
        """Test BlaxelExecutor instantiation with mocked Blaxel SDK."""

        # patch manually for Python 3.10 compatibility
        from unittest.mock import patch

        mod = importlib.import_module("blaxel.core.client.api.compute")
        patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = patcher.start()

        logger = MagicMock()
        mock_settings.headers = {}

        # Mock sandbox response
        mock_response = MagicMock()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(additional_imports=[], logger=logger)

        patcher.stop()

        assert executor.sandbox_name.startswith("smolagent-executor-")
        assert executor.image == "blaxel/jupyter-notebook"
        assert executor.memory == 4096
        assert executor.region is None

    @patch("smolagents.remote_executors.BlaxelExecutor.install_packages")
    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_custom_parameters(
        self, mock_settings, mock_sandbox_instance, mock_create_kernel, mock_install_packages
    ):
        """Test BlaxelExecutor with custom parameters."""
        logger = MagicMock()
        mock_settings.headers = {}
        mock_install_packages.return_value = ["numpy"]

        # Mock sandbox response
        mock_response = MagicMock()

        # patch manually for Python 3.10 compatibility
        mod = importlib.import_module("blaxel.core.client.api.compute")
        create_sandbox_patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = create_sandbox_patcher.start()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.us-was-1.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(
            additional_imports=["numpy"],
            logger=logger,
            sandbox_name="test-sandbox",
            image="custom-image:latest",
            memory=8192,
            region="us-was-1",
        )

        create_sandbox_patcher.stop()

        assert executor.sandbox_name == "test-sandbox"
        assert executor.image == "custom-image:latest"
        assert executor.memory == 8192
        assert executor.region == "us-was-1"
        assert mock_install_packages.called

    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_cleanup(self, mock_settings, mock_sandbox_instance, mock_create_kernel):
        """Test BlaxelExecutor cleanup method."""

        # patch manually for Python 3.10 compatibility
        from unittest.mock import patch

        mod = importlib.import_module("blaxel.core.client.api.compute")
        create_sandbox_patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = create_sandbox_patcher.start()
        delete_sandbox_patcher = patch.object(mod, "delete_sandbox")
        mock_delete_sandbox = delete_sandbox_patcher.start()

        logger = MagicMock()
        mock_settings.headers = {}

        # Mock sandbox response
        mock_response = MagicMock()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(additional_imports=[], logger=logger)

        # Test cleanup
        executor.cleanup()
        create_sandbox_patcher.stop()
        delete_sandbox_patcher.stop()

        # Verify that delete_sandbox.sync was called
        assert mock_delete_sandbox.sync.called
        # Verify sandbox reference was cleaned up
        assert not hasattr(executor, "sandbox")
