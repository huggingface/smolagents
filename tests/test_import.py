import os
import subprocess
import tempfile


def test_import_smolagents_without_extras(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.run(["uv", "venv", venv_dir], check=True)

        # Install smolagents in the virtual environment
        subprocess.run(
            ["uv", "pip", "install", "--python", os.path.join(venv_dir, "bin", "python"), "smolagents @ ."], check=True
        )

        # Run the import test in the virtual environment
        result = subprocess.run(
            [os.path.join(venv_dir, "bin", "python"), "-c", "import smolagents"],
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
