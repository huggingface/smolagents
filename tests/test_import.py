import os
import subprocess
import tempfile


def _venv_python(venv_dir):
    """Path to a venv's Python, which lives in Scripts/ on Windows."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def test_venv_python_path(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert _venv_python("venv") == os.path.join("venv", "bin", "python")
    monkeypatch.setattr(os, "name", "nt")
    assert _venv_python("venv") == os.path.join("venv", "Scripts", "python.exe")


def test_import_smolagents_without_extras(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.run(["uv", "venv", venv_dir], check=True)

        # Install smolagents in the virtual environment
        subprocess.run(
            ["uv", "pip", "install", "--python", _venv_python(venv_dir), "smolagents @ ."], check=True
        )

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
