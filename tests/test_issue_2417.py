import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from smolagents.models import VLLMModel


def test_issue_2417(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"

    class FakeLLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.model_kwargs = kwargs
            self.sampling_params = None

        def generate(self, prompt, sampling_params):
            self.sampling_params = sampling_params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="4")])]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams
    tokenizers = ModuleType("vllm.tokenizers")
    tokenizers.get_tokenizer = lambda model_id: FakeTokenizer()
    transformers_utils = ModuleType("vllm.transformers_utils")
    transformers_utils.__path__ = []
    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams

    monkeypatch.setattr("smolagents.models._is_package_available", lambda package_name: package_name == "vllm")
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm,
            "vllm.tokenizers": tokenizers,
            "vllm.transformers_utils": transformers_utils,
            "vllm.sampling_params": sampling_params,
        },
    ):
        model = VLLMModel("test-model", max_tokens=4096)
        output = model.generate([{"role": "user", "content": "What is 2+2?"}])

    assert output.content == "4"
    assert model.model.sampling_params.kwargs["max_tokens"] == 4096
