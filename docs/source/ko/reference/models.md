# 모델[[models]]

<Tip warning={true}>

Smolagents는 언제든지 변경될 수 있는 실험적인 API입니다. 에이전트가 반환하는 결과는
API나 기반 모델이 변경될 수 있으므로 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 자세히 알아보려면 [소개 가이드](../index)를 꼭 읽어보세요. 이 페이지는
기반 클래스에 대한 API 문서를 포함하고 있습니다.

## 모델[[models]]

smolagents의 모든 모델 클래스는 인스턴스화 시점에 `temperature`, `max_tokens`, `top_p` 등과 같은 추가 키워드 인자를 직접 전달하는 것을 지원합니다.
이 파라미터들은 기반 모델의 생성(completion) 호출에 자동으로 전달되어, 창의성, 응답 길이, 샘플링 전략과 같은 모델의 동작을 설정할 수 있습니다.

### 기본 모델[[base-model]]

`Model` 클래스는 모든 모델 구현의 기반 역할을 하며, 사용자 정의 모델이 에이전트와 함께 작동하기 위해 구현해야 하는 핵심 인터페이스를 제공합니다.

[[autodoc]] Model

### API 모델[[api-model]]

`ApiModel` 클래스는 모든 API 기반 모델 구현의 기초 역할을 하며, 외부 API 상호 작용, 속도 제한, 그리고 API별 모델이 상속하는 클라이언트 관리를 위한 공통 기능을 제공합니다.

[[autodoc]] ApiModel

### TransformersModel[[transformersmodel]]

편의를 위해, 초기화 시 주어진 model_id에 대한 로컬 `transformers` 파이프라인을 구축하여 위 사항들을 구현하는 `TransformersModel`을 추가했습니다.

```python
from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}], stop_sequences=["great"]))
``````text
>>> What a
```

기반 모델에서 지원하는 모든 키워드 인자(`temperature`, `max_new_tokens`, `top_p` 등)를 인스턴스화 시점에 직접 전달할 수 있습니다. 이들은 모델 생성 호출로 전달됩니다:

```python
model = TransformersModel(
    model_id="HuggingFaceTB/SmolLM-135M-Instruct",
    temperature=0.7,
    max_new_tokens=1000
)
```

> [!TIP]
> 사용자의 컴퓨터에 `transformers`와 `torch`가 설치되어 있어야 합니다. 설치되지 않은 경우 `pip install 'smolagents[transformers]'`를 실행하세요.

[[autodoc]] TransformersModel

### InferenceClientModel[[inferenceclientmodel]]

`InferenceClientModel`은 LLM 실행을 위해 huggingface_hub의 [InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference)를 래핑합니다. 이는 Hub에서 사용 가능한 모든 [Inference Providers](https://huggingface.co/docs/inference-providers/index)를 지원합니다: Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together 등.

또한 `requests_per_minute` 인자를 사용하여 분당 요청 수로 속도 제한을 설정할 수 있습니다:

```python
from smolagents import InferenceClientModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = InferenceClientModel(provider="novita", requests_per_minute=60)
print(model(messages))
``````text
>>> Of course! If you change your mind, feel free to reach out. Take care!
```

기반 모델에서 지원하는 모든 키워드 인자(`temperature`, `max_tokens`, `top_p` 등)를 인스턴스화 시점에 직접 전달할 수 있습니다. 이들은 모델 생성 호출로 전달됩니다:

```python
model = InferenceClientModel(
    provider="novita",
    requests_per_minute=60,
    temperature=0.8,
    max_tokens=500
)
```

[[autodoc]] InferenceClientModel

### LiteLLMModel[[litellmmodel]]

`LiteLLMModel`은 [LiteLLM](https://www.litellm.ai/)을 활용하여 다양한 제공업체의 100개 이상의 LLM을 지원합니다.
모델 초기화 시 kwargs를 전달하여 모델을 사용할 때마다 해당 설정을 적용할 수 있습니다. 예를 들어 아래에서는 `temperature`를 전달합니다. 또한 `requests_per_minute` 인자를 사용하여 분당 요청 수로 속도 제한을 설정할 수 있습니다.

```python
from smolagents import LiteLLMModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=10, requests_per_minute=60)
print(model(messages))
```

[[autodoc]] LiteLLMModel

### LiteLLMRouterModel[[litellmroutermodel]]

`LiteLLMRouterModel`은 [LiteLLM Router](https://docs.litellm.ai/docs/routing)를 감싼 래퍼(wrapper)로,
여러 배포 환경에 걸친 로드 밸런싱, 큐잉을 통한 중요 요청 우선 처리,
그리고 쿨다운, 폴백, 지수적 백오프 재시도와 같은 기본적인 신뢰성 조치 구현 등 고급 라우팅 전략을 활용합니다.

```python
from smolagents import LiteLLMRouterModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMRouterModel(
    model_id="llama-3.3-70b",
    model_list=[
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {"model": "groq/llama-3.3-70b", "api_key": os.getenv("GROQ_API_KEY")},
        },
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {"model": "cerebras/llama-3.3-70b", "api_key": os.getenv("CEREBRAS_API_KEY")},
        },
    ],
    client_kwargs={
        "routing_strategy": "simple-shuffle",
    },
)
print(model(messages))
```

[[autodoc]] LiteLLMRouterModel

### OpenAIServerModel[[openaiservermodel]]

이 클래스를 사용하면 OpenAIServer와 호환되는 모든 모델을 호출할 수 있습니다.
설정 방법은 다음과 같습니다 (`api_base` url을 다른 서버를 가리키도록 사용자 정의할 수 있습니다):
```py
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

기반 모델에서 지원하는 모든 키워드 인자(`temperature`, `max_tokens`, `top_p` 등)를 인스턴스화 시점에 직접 전달할 수 있습니다. 이들은 모델 생성 호출로 전달됩니다:

```py
model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
)
```

[[autodoc]] OpenAIServerModel

### AzureOpenAIServerModel[[azureopenaiservermodel]]

`AzureOpenAIServerModel`을 사용하면 모든 Azure OpenAI 배포에 연결할 수 있습니다.

아래에서 설정 예시를 확인할 수 있습니다. `azure_endpoint`, `api_key`, `api_version` 인자는 해당 환경 변수(`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`)를 설정했다면 생략할 수 있습니다.

`OPENAI_API_VERSION`에 `AZURE_` 접두사가 없는 점에 유의하세요. 이는 기반이 되는 [openai](https://github.com/openai/openai-python) 패키지의 설계 방식 때문입니다.

```py
import os

from smolagents import AzureOpenAIServerModel

model = AzureOpenAIServerModel(
    model_id = os.environ.get("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("OPENAI_API_VERSION")    
)
```

[[autodoc]] AzureOpenAIServerModel

### AmazonBedrockServerModel[[amazonbedrockservermodel]]

`AmazonBedrockServerModel`은 Amazon Bedrock에 연결하고 사용 가능한 모든 모델로 에이전트를 실행하는 데 도움을 줍니다.

아래는 설정 예시입니다. 이 클래스는 사용자 정의를 위한 추가 옵션도 제공합니다.

```py
import os

from smolagents import AmazonBedrockServerModel

model = AmazonBedrockServerModel(
    model_id = os.environ.get("AMAZON_BEDROCK_MODEL_ID"),
)
```

[[autodoc]] AmazonBedrockServerModel

### MLXModel[[mlxmodel]]


```python
from smolagents import MLXModel

model = MLXModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": "Ok!"}], stop_sequences=["great"]))
``````text
>>> What a
```

> [!TIP]
> 사용자의 컴퓨터에 `mlx-lm`이 설치되어 있어야 합니다. 설치되지 않은 경우 `pip install 'smolagents[mlx-lm]'`를 실행하세요.

[[autodoc]] MLXModel

### VLLMModel[[vllmmodel]]

빠른 LLM 추론 및 서빙을 위해 [vLLM](https://docs.vllm.ai/)을 사용하는 모델입니다.

```python
from smolagents import VLLMModel

model = VLLMModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": "Ok!"}], stop_sequences=["great"]))
```

> [!TIP]
> 사용자의 컴퓨터에 `vllm`이 설치되어 있어야 합니다. 설치되지 않은 경우 `pip install 'smolagents[vllm]'`를 실행하세요.

[[autodoc]] VLLMModel

### 사용자 정의 모델[[custom-model]]

자유롭게 자신만의 모델을 만들어 에이전트를 구동하는 데 사용할 수 있습니다.

기본 `Model` 클래스를 서브클래싱하여 에이전트를 위한 모델을 만들 수 있습니다.
주요 기준은 `generate` 메서드를 서브클래싱하는 것이며, 다음 두 가지 기준을 따릅니다:
1. 입력 `messages`에 대해 [메시지 형식](./chat_templating)(`List[Dict[str, str]]`)을 따르고, `.content` 속성을 가진 객체를 반환합니다.
2. `stop_sequences` 인자로 전달된 시퀀스에서 출력 생성을 중단합니다.

LLM을 정의하기 위해, 기본 `Model` 클래스를 상속하는 `CustomModel` 클래스를 만들 수 있습니다.
이 클래스는 [메시지](./chat_templating) 리스트를 받아 텍스트를 포함하는 `.content` 속성을 가진 객체를 반환하는 `generate` 메서드를 가져야 합니다. `generate` 메서드는 또한 생성을 중단할 시점을 나타내는 `stop_sequences` 인자를 받아들여야 합니다.

```python
from huggingface_hub import login, InferenceClient

from smolagents import Model

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

model_id = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(model=model_id)

class CustomModel(Model):
    def generate(messages, stop_sequences=["Task"]):
        response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1024)
        answer = response.choices[0].message
        return answer

custom_model = CustomModel()
```

또한, `generate`는 `grammar` 인자를 받아 [제약된 생성(constrained generation)](https://huggingface.co/docs/text-generation-inference/conceptual/guidance)을 허용하여 올바른 형식의 에이전트 출력을 강제할 수 있습니다.