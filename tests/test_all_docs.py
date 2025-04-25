# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import ast
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import pytest
from dotenv import load_dotenv

from .utils.markers import require_run_all


class SubprocessCallException(Exception):
    pass


def run_command(command: list[str], return_stdout=False, env=None):
    """
    Runs command with subprocess.check_output and returns stdout if requested.
    Properly captures and handles errors during command execution.
    """
    for i, c in enumerate(command):
        if isinstance(c, Path):
            command[i] = str(c)

    if env is None:
        env = os.environ.copy()

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


class DocCodeExtractor:
    """Handles extraction and validation of Python code from markdown files."""

    @staticmethod
    def extract_python_code(content: str) -> list[str]:
        """Extract Python code blocks from markdown content."""
        pattern = r"```(?:python|py)\n(.*?)\n```"
        matches = re.finditer(pattern, content, re.DOTALL)
        return [match.group(1).strip() for match in matches]

    @staticmethod
    def create_test_script(code_blocks: list[str], tmp_dir: str) -> Path:
        """Create a temporary Python script from code blocks."""
        combined_code = "\n\n".join(code_blocks)
        assert len(combined_code) > 0, "Code is empty!"
        tmp_file = Path(tmp_dir) / "test_script.py"

        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(combined_code)

        return tmp_file


# Skip: slow tests + require API keys
@require_run_all
class TestDocs:
    """Test case for documentation code testing."""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls.launch_args = ["python3"]
        cls.docs_dir = Path(__file__).parent.parent / "docs" / "source" / "en"
        cls.extractor = DocCodeExtractor()

        if not cls.docs_dir.exists():
            raise ValueError(f"Docs directory not found at {cls.docs_dir}")

        load_dotenv()

        cls.md_files = list(cls.docs_dir.rglob("*.mdx"))
        if not cls.md_files:
            raise ValueError(f"No markdown files found in {cls.docs_dir}")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir)

    @pytest.mark.timeout(100)
    def test_single_doc(self, doc_path: Path):
        """Test a single documentation file."""
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        code_blocks = self.extractor.extract_python_code(content)
        excluded_snippets = [
            "ToolCollection",
            "image_generation_tool",  # We don't want to run this expensive operation
            "from_langchain",  # Langchain is not a dependency
            "while llm_should_continue(memory):",  # This is pseudo code
            "ollama_chat/llama3.2",  # Exclude ollama building in guided tour
            "model = TransformersModel(model_id=model_id)",  # Exclude testing with transformers model
            "SmolagentsInstrumentor",  # Exclude telemetry since it needs additional installs
        ]
        code_blocks = [
            block
            for block in code_blocks
            if not any(
                [snippet in block for snippet in excluded_snippets]
            )  # Exclude these tools that take longer to run and add dependencies
        ]
        if len(code_blocks) == 0:
            pytest.skip(f"No Python code blocks found in {doc_path.name}")

        # Validate syntax of each block individually by parsing it
        for i, block in enumerate(code_blocks, 1):
            ast.parse(block)

        # Create and execute test script
        print("\n\nCollected code block:==========\n".join(code_blocks))
        try:
            code_blocks = [
                (
                    block.replace("<YOUR_HUGGINGFACEHUB_API_TOKEN>", os.getenv("HF_TOKEN"))
                    .replace("YOUR_ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
                    .replace("{your_username}", "m-ric")
                )
                for block in code_blocks
            ]
            test_script = self.extractor.create_test_script(code_blocks, self._tmpdir)
            run_command(self.launch_args + [str(test_script)])

        except SubprocessCallException as e:
            pytest.fail(f"\nError while testing {doc_path.name}:\n{str(e)}")
        except Exception:
            pytest.fail(f"\nUnexpected error while testing {doc_path.name}:\n{traceback.format_exc()}")

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Fixture to ensure temporary directory exists for each test."""
        os.makedirs(self._tmpdir, exist_ok=True)
        yield
        # Clean up test files after each test
        for file in Path(self._tmpdir).glob("*"):
            file.unlink()


def pytest_generate_tests(metafunc):
    """Generate test cases for each markdown file."""
    if "doc_path" in metafunc.fixturenames:
        test_class = metafunc.cls

        # Initialize the class if needed
        if not hasattr(test_class, "md_files"):
            test_class.setup_class()

        # Parameterize with the markdown files
        metafunc.parametrize("doc_path", test_class.md_files, ids=[f.stem for f in test_class.md_files])
