import importlib.util
import pathlib

from datasets import Dataset
from test_agents import FakeCodeModel

from smolagents.tools import Tool


spec = importlib.util.spec_from_file_location(
    "dec_run",
    pathlib.Path(__file__).resolve().parents[1] / "examples" / "decentralized_smolagents_benchmark copy" / "run.py",
)
dec_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec_run)


class DummyTool(Tool):
    def __init__(self, name: str):
        self.name = name
        self.description = "dummy"
        self.inputs = {}
        self.output_type = "string"

    def forward(self) -> str:
        return "dummy"


def test_decentralized_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dec_run.datasets, "get_dataset_config_names", lambda ds: ["gaia", "math", "simpleqa"])
    fake_datasets = {
        "gaia": Dataset.from_dict({"question": ["red planet?"], "true_answer": ["Mars"], "source": ["GAIA"]}),
        "math": Dataset.from_dict({"question": ["2+2?"], "true_answer": ["4"], "source": ["MATH"]}),
        "simpleqa": Dataset.from_dict(
            {"question": ["author of 1984?"], "true_answer": ["George Orwell"], "source": ["SimpleQA"]}
        ),
    }
    monkeypatch.setattr(dec_run.datasets, "load_dataset", lambda dataset, task, split=None: fake_datasets[task])
    monkeypatch.setattr(dec_run, "GoogleSearchTool", lambda provider="serper": DummyTool("search"))
    monkeypatch.setattr(dec_run, "VisitWebpageTool", lambda *args, **kwargs: DummyTool("visit"))

    class DummyModel(FakeCodeModel):
        def __init__(self):
            super().__init__(model_id="dummy-model")

    monkeypatch.setattr(dec_run, "InferenceClientModel", lambda model_id, provider=None, max_tokens=8192: DummyModel())

    eval_ds = dec_run.load_eval_dataset("dummy", num_examples=1)
    model = dec_run.InferenceClientModel("dummy-model")
    dec_run.answer_questions(eval_ds, model, date="2025-01-01", parallel_workers=1, output_dir=str(tmp_path))

    for task in ["gaia", "math", "simpleqa"]:
        file = tmp_path / f"{model.model_id.replace('/', '__')}__code__{task}__2025-01-01.jsonl"
        assert file.exists()
