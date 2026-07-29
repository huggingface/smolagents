from pathlib import Path

import smolagents


def test_issue_2349():
    project_root = Path(smolagents.__file__).resolve().parents[2]
    tools_tutorial = (project_root / "docs/source/en/tutorials/tools.md").read_text()

    assert "https://huggingface.co/spaces/m-ric/hf-model-downloads" not in tools_tutorial
    assert "https://huggingface.co/spaces/mjschock/hf-model-downloads" in tools_tutorial
