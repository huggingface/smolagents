# 다양한 모델 사용하기 [[using-different-models]]

[[open-in-colab]]

`smolagents`는 다양한 프로바이더의 여러 언어 모델을 사용할 수 있는 유연한 프레임워크를 제공합니다.
이 가이드는 에이전트와 함께 다양한 모델 유형을 사용하는 방법을 보여줍니다.

## 사용 가능한 모델 유형 [[available-model-types]]

`smolagents`는 기본적으로 여러 모델 유형을 지원합니다:
1. [`InferenceClientModel`]: Hugging Face의 추론 API를 사용하여 모델에 접근
2. [`TransformersModel`]: 🤗 Transformers 라이브러리를 사용하여 로컬에서 모델 실행
3. [`VLLMModel`]: 최적화된 서빙으로 빠른 추론을 위해 vLLM 사용
4. [`MLXModel`]: MLX를 사용하여 Apple Silicon 디바이스에 최적화
5. [`LiteLLMModel`]: LiteLLM을 통해 수백 개의 대규모 언어 모델에 접근 제공
6. [`LiteLLMRouterModel`]: 여러 모델 간에 요청을 분산
7. [`OpenAIModel`]: OpenAI 호환 API를 구현하는 모든 프로바이더에 접근 제공
8. [`AzureOpenAIModel`]: Azure의 OpenAI 서비스 사용
9. [`AmazonBedrockModel`]: AWS Bedrock의 API에 연결

모든 모델 클래스는 인스턴스화 시점에 추가 키워드 인수들(`temperature`, `max_tokens`, `top_p` 등)을 직접 전달하는 것을 지원합니다.
이러한 매개변수들은 자동으로 기본 모델의 완성 호출로 전달되어, 창의성, 응답 길이, 샘플링 전략과 같은 모델 동작을 구성할 수 있게 해줍니다.

## Google Gemini 모델 사용하기 [[using-google-gemini-models]]

Google Gemini API 문서(https://ai.google.dev/gemini-api/docs/openai)에서 설명한 바와 같이,
Google은 Gemini 모델에 대해 OpenAI 호환 API를 제공하므로, 적절한 베이스 URL을 설정하여
[`OpenAIModel`]을 Gemini 모델과 함께 사용할 수 있습니다.

먼저, 필요한 의존성을 설치합니다:
```bash
pip install 'smolagents[openai]'
```

그다음, [Gemini API 키를 얻고](https://ai.google.dev/gemini-api/docs/api-key) 코드에서 설정합니다:
```python
GEMINI_API_KEY = <YOUR-GEMINI-API-KEY>
```

이제 `OpenAIModel` 클래스를 사용하고 `api_base` 매개변수를 Gemini API 베이스 URL로 설정하여
Gemini 모델을 초기화할 수 있습니다:
```python
from smolagents import OpenAIModel

model = OpenAIModel(
    model_id="gemini-2.0-flash",
    # Google Gemini OpenAI 호환 API 베이스 URL
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)
```

## OpenRouter 모델 사용하기 [[using-openrouter-models]]

OpenRouter는 통합된 OpenAI 호환 API를 통해 다양한 언어 모델에 대한 접근을 제공합니다.
적절한 베이스 URL을 설정하여 [`OpenAIModel`]을 사용해 OpenRouter에 연결할 수 있습니다.

먼저, 필요한 의존성을 설치합니다:
```bash
pip install 'smolagents[openai]'
```

그다음, [OpenRouter API 키를 얻고](https://openrouter.ai/keys) 코드에서 설정합니다:
```python
OPENROUTER_API_KEY = <YOUR-OPENROUTER-API-KEY>
```

이제 `OpenAIModel` 클래스를 사용하여 OpenRouter에서 사용 가능한 모든 모델을 초기화할 수 있습니다:
```python
from smolagents import OpenAIModel

model = OpenAIModel(
    # OpenRouter에서 사용 가능한 모든 모델 ID를 사용할 수 있습니다
    model_id="openai/gpt-4o",
    # OpenRouter API 베이스 URL
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```

## xAI의 Grok 모델 사용하기 [[using-xais-grok-models]]

xAI의 Grok 모델은 [`LiteLLMModel`]을 통해 접근할 수 있습니다.

일부 모델("grok-4" 및 "grok-3-mini" 등)은 `stop` 매개변수를 지원하지 않으므로,
API 호출에서 이를 제외하기 위해 `REMOVE_PARAMETER`를 사용해야 합니다.

먼저, 필요한 의존성을 설치합니다:
```bash
pip install smolagents[litellm]
```

그다음, [xAI API 키를 얻고](https://console.x.ai/) 코드에서 설정합니다:
```python
XAI_API_KEY = <YOUR-XAI-API-KEY>
```

이제 `LiteLLMModel` 클래스를 사용하여 Grok 모델을 초기화하고 해당되는 경우 `stop` 매개변수를 제거할 수 있습니다:
```python
from smolagents import LiteLLMModel, REMOVE_PARAMETER

# Grok-4 사용
model = LiteLLMModel(
    model_id="xai/grok-4",
    api_key=XAI_API_KEY,
    stop=REMOVE_PARAMETER,  # grok-4 모델이 이를 지원하지 않으므로 stop 매개변수 제거
    temperature=0.7
)

# 또는 Grok-3-mini 사용
model_mini = LiteLLMModel(
    model_id="xai/grok-3-mini",
    api_key=XAI_API_KEY,
    stop=REMOVE_PARAMETER,  # grok-3-mini 모델이 이를 지원하지 않으므로 stop 매개변수 제거
    max_tokens=1000
)
```
