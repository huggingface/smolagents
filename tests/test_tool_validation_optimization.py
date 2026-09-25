import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.parametrize("optimization_flag", ["-O", "-OO"])
def test_tool_validation_survives_python_optimization(optimization_flag):
    code = dedent(
        """
        from smolagents import Tool


        def expect_error(tool_cls, error_type, message):
            try:
                tool_cls()
            except error_type as exc:
                if message not in str(exc):
                    raise RuntimeError(
                        f"Expected error containing {message!r}, got {str(exc)!r}"
                    ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Expected {error_type.__name__}, got {type(exc).__name__}: {exc}"
                ) from exc
            else:
                raise RuntimeError(
                    f"Expected {error_type.__name__} for {tool_cls.__name__}, but no error was raised"
                )


        class InvalidInputContainerTool(Tool):
            name = "invalid_input_container"
            description = "Tool with a non-dictionary input specification."
            inputs = {"query": "not-a-dictionary"}
            output_type = "string"

            def forward(self, query: str) -> str:
                return query


        class MissingDescriptionTool(Tool):
            name = "missing_description"
            description = "Tool whose input schema is missing a description."
            inputs = {"query": {"type": "string"}}
            output_type = "string"

            def forward(self, query: str) -> str:
                return query


        class InvalidOutputTypeTool(Tool):
            name = "invalid_output_type"
            description = "Tool with an unsupported output type."
            inputs = {"query": {"type": "string", "description": "Input query."}}
            output_type = "unsupported"

            def forward(self, query: str) -> str:
                return query


        class NullableInputsOnlyTool(Tool):
            name = "nullable_inputs_only"
            description = "Tool whose nullable declaration disagrees with its signature."
            inputs = {
                "query": {
                    "type": "string",
                    "description": "Input query.",
                    "nullable": True,
                }
            }
            output_type = "string"

            def forward(self, query: str) -> str:
                return query


        class NullableSignatureOnlyTool(Tool):
            name = "nullable_signature_only"
            description = "Tool whose nullable signature disagrees with its input schema."
            inputs = {"query": {"type": "string", "description": "Input query."}}
            output_type = "string"

            def forward(self, query: str | None) -> str:
                return query or ""


        expect_error(InvalidInputContainerTool, TypeError, "should be a dictionary")
        expect_error(
            MissingDescriptionTool,
            ValueError,
            "should have keys 'type' and 'description'",
        )
        expect_error(InvalidOutputTypeTool, ValueError, "output_type")
        expect_error(
            NullableInputsOnlyTool,
            ValueError,
            "Nullable argument 'query' in inputs",
        )
        expect_error(
            NullableSignatureOnlyTool,
            ValueError,
            "Nullable argument 'query' in function signature",
        )
        """
    )

    result = subprocess.run(
        [sys.executable, optimization_flag, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
