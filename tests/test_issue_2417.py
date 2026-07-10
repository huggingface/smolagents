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

import smolagents.models
from smolagents.models import VLLMModel


def test_issue_2417(monkeypatch):
    created_llms = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "What is 2+2?"}]
            return "rendered prompt"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOutput:
        text = "4"
        token_ids = [1]

    class FakeGeneration:
        prompt_token_ids = [1, 2]
        outputs = [FakeOutput()]

    class FakeLLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.generate_kwargs = None
            self.sampling_params = None
            created_llms.append(self)

        def generate(self, prompt, sampling_params, **kwargs):
            assert prompt == "rendered prompt"
            assert "max_tokens" not in kwargs
            assert sampling_params.kwargs["max_tokens"] == 4096
            self.generate_kwargs = kwargs
            self.sampling_params = sampling_params
            return [FakeGeneration()]

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = FakeLLM
    vllm_module.SamplingParams = FakeSamplingParams

    vllm_sampling_params_module = types.ModuleType("vllm.sampling_params")
    vllm_sampling_params_module.StructuredOutputsParams = object

    vllm_tokenizers_module = types.ModuleType("vllm.tokenizers")
    vllm_tokenizers_module.get_tokenizer = lambda model_id: FakeTokenizer()

    for module_name in [
        "vllm",
        "vllm.sampling_params",
        "vllm.tokenizers",
        "vllm.transformers_utils",
        "vllm.transformers_utils.tokenizer",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", vllm_sampling_params_module)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", vllm_tokenizers_module)
    monkeypatch.setattr(smolagents.models, "_is_package_available", lambda package_name: package_name == "vllm")

    model = VLLMModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_kwargs={"max_model_len": 2048, "dtype": "float16"},
        max_tokens=4096,
    )

    output = model.generate([{"role": "user", "content": "What is 2+2?"}])

    assert output.content == "4"
    assert created_llms[0].kwargs == {"max_model_len": 2048, "dtype": "float16"}
