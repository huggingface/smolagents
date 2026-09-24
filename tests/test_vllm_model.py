import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smolagents.models import ChatMessage, MessageRole, VLLMModel


def _mock_vllm_module_stack(use_primary_tokenizer: bool, case_id: int):
    model_output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(text="ok", token_ids=[1, 2, 3])])
    vllm_instance = MagicMock()
    vllm_instance.generate.return_value = [model_output]
    llm_ctor = MagicMock(return_value=vllm_instance)

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    primary_tokenizer = MagicMock(name=f"primary_tokenizer_{case_id}")
    fallback_tokenizer = MagicMock(name=f"fallback_tokenizer_{case_id}")
    primary_tokenizer.apply_chat_template.return_value = "primary prompt"
    fallback_tokenizer.apply_chat_template.return_value = "fallback prompt"

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = llm_ctor
    vllm_module.SamplingParams = _FakeSamplingParams

    transformers_utils = types.ModuleType("vllm.transformers_utils")
    transformers_tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizer")
    if use_primary_tokenizer:
        transformers_tokenizer_module.get_tokenizer = MagicMock(return_value=primary_tokenizer)
    transformers_utils.tokenizer = transformers_tokenizer_module

    fallback_tokens = types.ModuleType("vllm.tokenizers")
    fallback_tokens.get_tokenizer = MagicMock(return_value=fallback_tokenizer)

    vllm_sampling_module = types.ModuleType("vllm.sampling_params")
    vllm_sampling_module.StructuredOutputsParams = _FakeStructuredOutputsParams

    vllm_module.transformers_utils = transformers_utils
    vllm_module.tokenizers = fallback_tokens

    return {
        "vllm": vllm_module,
        "vllm.transformers_utils": transformers_utils,
        "vllm.transformers_utils.tokenizer": transformers_tokenizer_module,
        "vllm.tokenizers": fallback_tokens,
        "vllm.sampling_params": vllm_sampling_module,
    }, llm_ctor, vllm_instance, primary_tokenizer, fallback_tokenizer


@pytest.mark.parametrize("case_id, use_primary_tokenizer", [(i, i % 2 == 0) for i in range(200)])
def test_vllmmodel_tokenizer_compat(case_id, use_primary_tokenizer):
    modules, llm_ctor, vllm_instance, primary_tokenizer, fallback_tokenizer = _mock_vllm_module_stack(
        use_primary_tokenizer=use_primary_tokenizer,
        case_id=case_id,
    )

    model_id = f"compat-model-{case_id}"
    with patch.dict("sys.modules", modules):
        with patch("smolagents.models._is_package_available", return_value=True):
            model = VLLMModel(model_id=model_id)

    if use_primary_tokenizer:
        assert model.tokenizer is primary_tokenizer
        assert modules["vllm.transformers_utils.tokenizer"].get_tokenizer.call_count == 1
        assert modules["vllm.tokenizers"].get_tokenizer.call_count == 0
    else:
        assert model.tokenizer is fallback_tokenizer
        assert modules["vllm.transformers_utils.tokenizer"].__dict__.get("get_tokenizer", None) is None
        assert modules["vllm.tokenizers"].get_tokenizer.call_count == 1

    llm_ctor.assert_called_once_with(model=model_id)
    assert model.model is vllm_instance


TOKENIZER_CASES = []
for i in range(200):
    if i % 4 == 0:
        init = None
        runtime = None
        expected = 2048
    elif i % 4 == 1:
        init = 64 + i
        runtime = None
        expected = init
    elif i % 4 == 2:
        init = None
        runtime = 128 + i
        expected = runtime
    else:
        init = 256 + i
        runtime = 64 + i
        expected = runtime
    TOKENIZER_CASES.append((init, runtime, expected))


@pytest.mark.parametrize("init_tokens, runtime_tokens, expected_tokens", TOKENIZER_CASES)
def test_vllmmodel_respects_max_tokens_and_removes_kwarg_from_generate(init_tokens, runtime_tokens, expected_tokens):
    modules, _, vllm_instance, _, _ = _mock_vllm_module_stack(use_primary_tokenizer=True, case_id=init_tokens or 0)
    fallback = MagicMock()
    fallback.apply_chat_template = MagicMock(return_value="prompt")
    modules["vllm.transformers_utils.tokenizer"].get_tokenizer = MagicMock(return_value=fallback)
    modules["vllm.tokenizers"].get_tokenizer = MagicMock(return_value=fallback)

    init_kwargs = {}
    if init_tokens is not None:
        init_kwargs["max_tokens"] = init_tokens

    runtime_kwargs = {}
    if runtime_tokens is not None:
        runtime_kwargs["max_tokens"] = runtime_tokens

    messages = [
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "How are you?"}]),
    ]

    with patch.dict("sys.modules", modules):
        with patch("smolagents.models._is_package_available", return_value=True):
            model = VLLMModel(model_id=f"max-tokens-{init_tokens}-{runtime_tokens}", **init_kwargs)
            response = model(messages=messages, **runtime_kwargs)

    assert response.content == "ok"
    generate_call = vllm_instance.generate.call_args
    assert "max_tokens" not in generate_call.kwargs
    sampling_params = generate_call.kwargs["sampling_params"]
    assert sampling_params.max_tokens == expected_tokens
