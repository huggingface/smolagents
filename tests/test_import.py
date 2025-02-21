import subprocess


def test_import_smolagents_without_extras():
    # Run the import statement in an isolated virtual environment
    result = subprocess.run(
        ["uv", "run", "--isolated", "--no-editable", "-"], input="import smolagents", text=True, capture_output=True
    )
    # Check if the import was successful
    assert result.returncode == 0, (
        "Import failed with error: "
        + (result.stderr.splitlines()[-1] if result.stderr else "No error message")
        + "\n"
        + result.stderr
    )
