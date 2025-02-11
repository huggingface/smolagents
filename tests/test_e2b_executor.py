from unittest.mock import MagicMock, patch

from smolagents.e2b_executor import E2BExecutor


class TestE2BExecutor:
    def test_e2b_executor_instantiation(self):
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            mock_sandbox.return_value.commands.run.return_value.error = None
            mock_sandbox.return_value.run_code.return_value.error = None
            executor = E2BExecutor(additional_imports=[], tools=[], logger=logger)
        assert isinstance(executor, E2BExecutor)
        assert executor.logger == logger
        assert executor.final_answer is False
        assert executor.custom_tools == {}
        assert executor.final_answer_pattern.pattern == r"final_answer\((.*?)\)"
        assert executor.sbx == mock_sandbox.return_value
