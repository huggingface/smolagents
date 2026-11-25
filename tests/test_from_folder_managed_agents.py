"""from_folder must keep an explicit managed_agents override."""

from pathlib import Path
from tempfile import TemporaryDirectory

from smolagents import CodeAgent, OpenAIModel


def test_explicit_managed_agents_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "local-placeholder")
    model = OpenAIModel(
        model_id="local",
        api_key="local-placeholder",
        api_base="http://127.0.0.1:1/v1",
    )
    saved_worker = CodeAgent(
        tools=[],
        model=model,
        name="worker",
        description="Saved worker.",
        max_steps=1,
        verbosity_level=0,
    )
    parent = CodeAgent(
        tools=[], model=model, managed_agents=[saved_worker], verbosity_level=0
    )
    replacement = CodeAgent(
        tools=[],
        model=model,
        name="worker",
        description="Replacement worker.",
        max_steps=7,
        verbosity_level=0,
    )

    with TemporaryDirectory() as directory:
        folder = Path(directory)
        parent.save(folder)
        restored = CodeAgent.from_folder(
            folder, model=model, managed_agents=[replacement]
        )

    actual = restored.managed_agents["worker"]
    assert actual is replacement
    assert actual.max_steps == 7
