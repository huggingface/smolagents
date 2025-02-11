import subprocess
import sys


def test_import_smolagents_without_extras():
    # Create a new Python process to test the import
    cmd = [sys.executable, "-c", "import smolagents"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if the import was successful
    assert result.returncode == 0, (
        "Import failed with error: "
        + (result.stderr.splitlines()[-1] if result.stderr else "No error message")
        + "\n"
        + result.stderr
    )
