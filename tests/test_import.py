import os
import subprocess
import tempfile


def _venv_python(venv_dir: str) -> str:
    """Return the interpreter path created inside a virtual environment by `uv venv`."""
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv_dir, scripts_dir, executable)


def test_import_smolagents_without_extras(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.run(["uv", "venv", venv_dir], check=True)

        # Install smolagents in the virtual environment
        subprocess.run(["uv", "pip", "install", "--python", _venv_python(venv_dir), "smolagents @ ."], check=True)

        # Run the import test in the virtual environment
        result = subprocess.run(
            [_venv_python(venv_dir), "-c", "import smolagents"],
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
