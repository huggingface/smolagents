# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import types
from types import SimpleNamespace

import smolagents.models as models_module
from smolagents.models import VLLMModel


def test_issue_2417(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "What is 2+2?"}]
            return "What is 2+2?"

    class FakeLLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.last_sampling_params = None

        def generate(self, prompt, sampling_params):
            self.last_sampling_params = sampling_params
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(text="4", token_ids=[3])],
                )
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.max_tokens = kwargs["max_tokens"]

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_module.LLM = FakeLLM
    vllm_module.SamplingParams = FakeSamplingParams

    tokenizers_module = types.ModuleType("vllm.tokenizers")
    tokenizers_module.get_tokenizer = lambda model_id: FakeTokenizer()

    sampling_params_module = types.ModuleType("vllm.sampling_params")
    sampling_params_module.StructuredOutputsParams = FakeStructuredOutputsParams

    monkeypatch.setattr(models_module, "_is_package_available", lambda package_name: package_name == "vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers_module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params_module)
    monkeypatch.delitem(sys.modules, "vllm.transformers_utils", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.transformers_utils.tokenizer", raising=False)

    model = VLLMModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_kwargs={"max_model_len": 2048, "dtype": "float16"},
        max_tokens=4096,
    )

    output = model.generate([{"role": "user", "content": "What is 2+2?"}])

    assert output.content == "4"
    assert model.model.last_sampling_params.max_tokens == 4096
