import os
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent


def _venv_python(venv_dir):
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv_dir, scripts_dir, executable)


def test_import_smolagents_without_extras(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.run(["uv", "venv", venv_dir], check=True)
        venv_python = _venv_python(venv_dir)

        # Install smolagents in the virtual environment
        subprocess.run(["uv", "pip", "install", "--python", venv_python, "smolagents @ ."], check=True)

        # Run the import test in the virtual environment
        result = subprocess.run(
            [venv_python, "-c", "import smolagents"],
            capture_output=True,
            text=True,
        )

    # Check if the import was successful
    assert result.returncode == 0, (
        "Import failed with error: "
        + (result.stderr.splitlines()[-1] if result.stderr else "No error message")
        + "\n"
        + result.stderr
    )


def test_entry_point_executor_after_install(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    repository_dir = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        venv_dir = temp_path / "venv"
        plugin_dir = temp_path / "demo-executor"
        plugin_module_dir = plugin_dir / "src" / "demo_executor"
        plugin_module_dir.mkdir(parents=True)

        (plugin_dir / "pyproject.toml").write_text(
            dedent(
                """
                [build-system]
                requires = ["setuptools>=61"]
                build-backend = "setuptools.build_meta"

                [project]
                name = "demo-executor"
                version = "0.0.0"
                requires-python = ">=3.10"

                [project.entry-points."smolagents.executors"]
                demo = "demo_executor:DemoExecutor"

                [tool.setuptools.packages.find]
                where = ["src"]
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (plugin_module_dir / "__init__.py").write_text(
            dedent(
                """
                from smolagents import CodeOutput, PythonExecutor


                class DemoExecutor(PythonExecutor):
                    def __init__(self, additional_authorized_imports, logger, marker):
                        self.marker = marker

                    def send_tools(self, tools):
                        pass

                    def send_variables(self, variables):
                        pass

                    def __call__(self, code_action):
                        return CodeOutput(
                            output=f"{self.marker}:{code_action}", logs="", is_final_answer=False
                        )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        subprocess.run(["uv", "venv", str(venv_dir)], check=True)
        venv_python = _venv_python(str(venv_dir))
        subprocess.run(
            ["uv", "pip", "install", "--python", venv_python, "smolagents @ ."],
            cwd=repository_dir,
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", venv_python, "--no-deps", str(plugin_dir)],
            check=True,
        )

        result = subprocess.run(
            [
                venv_python,
                "-c",
                dedent(
                    """
                    from unittest.mock import MagicMock

                    from smolagents import CodeAgent


                    agent = CodeAgent(
                        tools=[],
                        model=MagicMock(),
                        executor_type="demo",
                        executor_kwargs={"marker": "installed"},
                    )
                    assert type(agent.python_executor).__name__ == "DemoExecutor"
                    output = agent.python_executor("deterministic-code")
                    assert output.output == "installed:deterministic-code"
                    print("installed entry point resolved and executed")
                    """
                ),
            ],
            cwd=repository_dir,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, f"E2E subprocess failed:\n{result.stdout}\n{result.stderr}"
    assert "installed entry point resolved and executed" in result.stdout
