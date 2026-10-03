# 에이전트 안내서[[agents---guided-tour]]

[[open-in-colab]]

이 안내서에서는 에이전트를 구축하는 방법, 실행하는 방법, 그리고 사용 사례에 맞게 더 잘 작동하도록 맞춤 설정하는 방법을 학습합니다.

## 에이전트 유형 선택: CodeAgent 또는 ToolCallingAgent[[choosing-an-agent-type:-codeagent-or-toolcallingagent]]

`smolagents`는 [`CodeAgent`]와 [`ToolCallingAgent`] 두 가지 에이전트 클래스를 제공하는데, 이 두 클래스는 각각 에이전트가 도구와 상호작용하는 방법이 다릅니다.
두 방식의 핵심 차이점은 '액션을 지정하고 실행'하는 방식에 있습니다: 코드 생성 vs 구조화된 도구 호출.

- [`CodeAgent`]는 도구 호출을 Python 코드 스니펫으로 생성합니다.
  - 코드는 로컬에서 실행되거나(잠재적으로 불안전) 보안 샌드박스에서 실행됩니다.
  - 도구는 Python 함수로 노출됩니다(바인딩을 통해).
  - 도구 호출 예시:
    ```py
    result = search_docs("What is the capital of France?")
    print(result)
    ```
  - 장점:
    - 높은 표현력: 복잡한 로직과 제어 흐름을 허용하고 도구를 결합하고, 반복하고, 변환하고, 추론할 수 있습니다.
    - 유연성: 모든 가능한 액션을 미리 정의할 필요가 없으며, 동적으로 새로운 액션/도구를 생성할 수 있습니다.
    - 창발적 추론: 다단계 문제나 동적 로직에 이상적입니다.
  - 제한사항
    - 오류 위험: 구문 오류, 예외를 처리해야 합니다.
    - 예측성 부족: 예상치 못한 또는 안전하지 않은 출력에 더 취약합니다.
    - 보안 실행 환경이 필요합니다.

- [`ToolCallingAgent`]는 도구 호출을 구조화된 JSON으로 작성합니다.
  - 이는 많은 프레임워크(OpenAI API)에서 사용되는 일반적인 형식으로, 코드 실행 없이 구조화된 도구 상호작용을 가능하게 합니다.
  - 도구는 JSON 스키마로 정의됩니다: 이름, 설명, 매개변수 타입 등.
  - 도구 호출 예시:
    ```json
    {
      "tool_call": {
        "name": "search_docs",
        "arguments": {
          "query": "What is the capital of France?"
        }
      }
    }
    ```
  - 장점:
    - 안정성: 환각이 적고, 출력이 구조화되고 검증됩니다.
    - 안전성: 인수가 엄격하게 검증되고, 임의의 코드가 실행될 위험이 없습니다.
    - 상호 운용성: 외부 API나 서비스에 쉽게 매핑됩니다.
  - 제한사항:
    - 낮은 표현력: 결과를 동적으로 쉽게 결합하거나 변환할 수 없고, 복잡한 로직이나 제어 흐름을 수행할 수 없습니다.
    - 유연성 부족: 모든 가능한 액션을 미리 정의해야 하고, 사전 정의된 도구로 제한됩니다.
    - 코드 합성 없음: 도구 기능으로 제한됩니다.

어떤 에이전트 유형을 사용할지:
- [`CodeAgent`]를 사용하는 경우:
  - 추론, 연결 또는 동적 구성이 필요한 경우.
  - 도구가 결합할 수 있는 함수인 경우(예: 구문 분석 + 수학 + 쿼리).
  - 에이전트가 문제 해결자 또는 프로그래머인 경우.

- [`ToolCallingAgent`]를 사용하는 경우:
  - 단순하고 독립적인 도구가 있는 경우(예: API 호출, 문서 가져오기).
  - 높은 안정성과 명확한 검증을 원하는 경우.
  - 에이전트가 디스패처나 컨트롤러 같은 역할인 경우.

## CodeAgent[[codeagent]]

[`CodeAgent`]는 액션을 수행하고 작업을 해결하기 위해 Python 코드 스니펫을 생성합니다.

기본적으로 Python 코드 실행은 로컬 환경에서 수행됩니다.
사용자가 제공한 도구들(특히 Hugging Face 도구만 있는 경우)과 `print`나 `math` 모듈 함수 같은 사전 정의된 안전한 함수들만 호출할 수 있도록 제한되어 있어 안전합니다.

Python 인터프리터는 기본적으로 안전 목록에 포함된 모듈만 import를 허용하므로, 대부분의 명백한 보안 공격을 방지할 수 있습니다.
[`CodeAgent`]를 초기화할 때 `additional_authorized_imports` 인수에 문자열 목록으로 승인된 모듈을 전달하여 추가 import를 승인할 수 있습니다:

```py
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

또한 추가 보안 계층으로, import 목록에서 명시적으로 승인되지 않는 한 서브모듈에 대한 접근은 기본적으로 금지됩니다.
예를 들어, `numpy.random` 서브모듈에 접근하려면 `additional_authorized_imports` 목록에 `'numpy.random'`을 추가해야 합니다.
이는 `numpy`와 `numpy.random` 같은 모든 서브패키지 및 자체 서브패키지를 허용하는 `numpy.*`를 사용하여 승인할 수도 있습니다.

> [!WARNING]
> LLM은 실행될 임의의 코드를 생성할 수 있습니다: 안전하지 않은 import는 추가하지 마세요!

불법적인 작업을 수행하려고 시도하는 코드나 에이전트가 생성한 코드에 일반적인 Python 오류가 있는 경우 실행이 중단됩니다.

로컬 Python 인터프리터 대신 [E2B code executor](https://e2b.dev/docs#what-is-e2-b)나 Docker를 사용할 수도 있습니다. E2B의 경우, 먼저 [`E2B_API_KEY` 환경 변수를 설정](https://e2b.dev/dashboard?tab=keys)한 다음 에이전트 초기화 시 `executor_type="e2b"`를 전달하세요. Docker의 경우, 초기화 중에 `executor_type="docker"`를 전달하세요.

> [!TIP]
> 코드 실행에 대해 더 자세히 알아보려면 [이 튜토리얼](tutorials/secure_code_execution)을 확인하세요.

### ToolCallingAgent[[toolcallingagent]]

[`ToolCallingAgent`]는 많은 프레임워크(OpenAI API)에서 사용되는 일반적인 형식인 JSON 도구 호출을 출력하여, 코드 실행 없이 구조화된 도구 상호작용을 가능하게 합니다.

코드를 실행하지 않으므로 `additional_authorized_imports` 없이도 [`CodeAgent`]와 거의 동일한 방식으로 작동합니다:

```py
from smolagents import ToolCallingAgent, WebSearchTool

agent = ToolCallingAgent(tools=[WebSearchTool()], model=model)
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

## 에이전트 구축[[building-your-agent]]

최소한의 에이전트를 초기화하려면 최소한 다음 두 인수가 필요합니다:

- `model`, 에이전트를 구동하는 텍스트 생성 모델 - 에이전트는 단순한 LLM과 다르며, LLM을 엔진으로 사용하는 시스템입니다. 다음 옵션 중 하나를 사용할 수 있습니다:
    - [`TransformersModel`]은 사전 초기화된 `transformers` 파이프라인을 가져와 `transformers`를 사용하여 로컬 머신에서 추론을 실행합니다.
    - [`InferenceClientModel`]은 내부적으로 `huggingface_hub.InferenceClient`를 활용하며 Hub의 모든 추론 제공자를 지원합니다: Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together 등.
    - [`LiteLLMModel`]은 마찬가지로 [LiteLLM](https://docs.litellm.ai/)을 통해 100개 이상의 다양한 모델과 제공자를 호출할 수 있습니다!
    - [`AzureOpenAIModel`]은 [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service)에 배포된 OpenAI 모델을 사용할 수 있게 해줍니다.
    - [`AmazonBedrockModel`]은 [AWS](https://aws.amazon.com/bedrock/?nc1=h_ls)의 Amazon Bedrock을 사용할 수 있게 해줍니다.
    - [`MLXModel`]은 로컬 머신에서 추론을 실행하기 위한 [mlx-lm](https://pypi.org/project/mlx-lm/) 파이프라인을 생성합니다.

- `tools`, 에이전트가 작업 해결에 사용할 수 있는 도구 목록입니다. 빈 목록으로 설정할 수도 있습니다. add_base_tools=True 옵션을 사용하면 기본 제공되는 도구들(웹 검색, 코드 실행, 음성 인식 등)을 `tools` 목록에 추가할 수 있습니다.

`tools`와 `model` 두 인수를 설정하면 에이전트를 생성하고 실행할 수 있습니다. [추론 제공자](https://huggingface.co/blog/inference-providers), [transformers](https://github.com/huggingface/transformers/), [ollama](https://ollama.com/), [LiteLLM](https://www.litellm.ai/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), [Amazon Bedrock](https://aws.amazon.com/bedrock/?nc1=h_ls), 또는 [mlx-lm](https://pypi.org/project/mlx-lm/)을 통해 원하는 LLM을 사용할 수 있습니다.

모든 모델 클래스는 인스턴스화 시점에 추가 키워드 인수(예: `temperature`, `max_tokens`, `top_p` 등)를 직접 전달하는 것을 지원합니다.
이러한 매개변수는 기본 모델의 완성 호출에 자동으로 전달되어 창의성, 응답 길이, 샘플링 전략 등의 모델 동작을 구성할 수 있습니다.

<hfoptions id="Pick a LLM">
<hfoption id="Inference Providers">

추론 제공자는 인증을 위해 `HF_TOKEN`이 필요하지만, 무료 HF 계정에는 이미 포함된 크레딧이 제공됩니다. PRO로 업그레이드하여 포함된 크레딧을 늘리세요.

제한된 모델에 접근하거나 PRO 계정으로 속도 제한을 높이려면 환경 변수 `HF_TOKEN`을 설정하거나 `InferenceClientModel` 초기화 시 `token` 변수를 전달해야 합니다. [설정 페이지](https://huggingface.co/settings/tokens)에서 토큰을 얻을 수 있습니다.

```python
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = InferenceClientModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>") # You can choose to not pass any model_id to InferenceClientModel to use a default model
# you can also specify a particular provider e.g. provider="together" or provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Local Transformers Model">

```python
# !pip install smolagents[transformers]
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="OpenAI or Anthropic API">

`LiteLLMModel`을 사용하려면 환경 변수 `ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`를 설정하거나 초기화 시 `api_key` 변수를 전달해야 합니다.

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY") # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Ollama">

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2", # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_key="YOUR_API_KEY", # replace with API key if necessary
    num_ctx=8192, # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Azure OpenAI">

Azure OpenAI에 연결하려면 `AzureOpenAIModel`을 직접 사용하거나 `LiteLLMModel`을 사용하여 적절히 구성할 수 있습니다.

`AzureOpenAIModel`의 인스턴스를 초기화하려면 모델 배포 이름을 전달한 다음 `azure_endpoint`, `api_key`, `api_version` 인수를 전달하거나 환경 변수 `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`을 설정해야 합니다.

```python
# !pip install smolagents[openai]
from smolagents import CodeAgent, AzureOpenAIModel

model = AzureOpenAIModel(model_id="gpt-4o-mini")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

마찬가지로 다음과 같이 `LiteLLMModel`을 구성하여 Azure OpenAI에 연결할 수 있습니다:

- 모델 배포 이름을 `model_id`로 전달하고, 앞에 `azure/`를 붙여야 합니다.
- 환경 변수 `AZURE_API_VERSION`을 설정해야 합니다.
- `api_base`와 `api_key` 인수를 전달하거나 환경 변수 `AZURE_API_KEY`, `AZURE_API_BASE`를 설정합니다.

```python
import os
from smolagents import CodeAgent, LiteLLMModel

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-35-turbo-16k-deployment" # example of deployment name

os.environ["AZURE_API_KEY"] = "" # api_key
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2024-10-01-preview"

model = LiteLLMModel(model_id="azure/" + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
   "Could you give me the 118th number in the Fibonacci sequence?",
)
```

</hfoption>
<hfoption id="Amazon Bedrock">

`AmazonBedrockModel` 클래스는 Amazon Bedrock과 직접 연동되어 API 호출과 세부 구성을 지원합니다.

기본 사용법:

```python
# !pip install smolagents[aws_sdk]
from smolagents import CodeAgent, AmazonBedrockModel

model = AmazonBedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

고급 구성:

```python
import boto3
from smolagents import AmazonBedrockModel

# Create a custom Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY'
)

additional_api_config = {
    "inferenceConfig": {
        "maxTokens": 3000
    },
    "guardrailConfig": {
        "guardrailIdentifier": "identify1",
        "guardrailVersion": 'v1'
    },
}

# Initialize with comprehensive configuration
model = AmazonBedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
    client=bedrock_client,  # Use custom client
    **additional_api_config
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

LiteLLMModel 사용:

또는 Bedrock 모델과 함께 `LiteLLMModel`을 사용할 수 있습니다:

```python
from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(model_name="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model)

agent.run("Explain the concept of quantum computing")
```

</hfoption>
<hfoption id="mlx-lm">

```python
# !pip install smolagents[mlx-lm]
from smolagents import CodeAgent, MLXModel

mlx_model = MLXModel("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
agent = CodeAgent(model=mlx_model, tools=[], add_base_tools=True)

agent.run("Could you give me the 118th number in the Fibonacci sequence?")
```

</hfoption>
</hfoptions>

## 고급 에이전트 구성[[advanced-agent-configuration]]

### 에이전트 종료 조건 맞춤 설정[[customizing-agent-termination-conditions]]

기본적으로 에이전트는 `final_answer` 함수를 호출하거나 최대 단계 수에 도달할 때까지 계속 실행됩니다.
`final_answer_checks` 매개변수는 에이전트가 실행을 종료하는 시점과 방법을 더 세밀하게 제어할 수 있게 해줍니다:

```python
from smolagents import CodeAgent, InferenceClientModel

# Define a custom final answer check function
def is_integer(final_answer: str, agent_memory=None) -> bool:
    """Return True if final_answer is an integer."""
    try:
        int(final_answer)
        return True
    except ValueError:
        return False

# Initialize agent with custom final answer check
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    final_answer_checks=[is_integer]
)

agent.run("Calculate the least common multiple of 3 and 7")
```

`final_answer_checks` 매개변수는 각각 다음과 같은 함수들의 목록을 받습니다:
- 에이전트의 final_answer 문자열과 에이전트의 메모리를 매개변수로 받습니다
- final_answer가 유효한지(True) 아닌지(False)를 나타내는 불리언을 반환합니다

함수 중 하나라도 `False`를 반환하면 에이전트는 오류 메시지를 로그에 기록하고 실행을 계속합니다.
이 검증 메커니즘은 다음을 가능하게 합니다:
- 출력 형식 요구사항 강제(예: 수학 문제에 대한 숫자 답변 보장)
- 도메인별 검증 규칙 구현
- 자체 출력을 검증하는 더 견고한 에이전트 생성

## 에이전트 실행 검사[[inspecting-an-agent-run]]

실행 후 무슨 일이 일어났는지 확인하는 데 유용한 몇 가지 속성이 있습니다:

- `agent.logs`는 에이전트의 상세한 실행 로그를 저장합니다. 에이전트 실행의 각 단계마다 모든 정보가 딕셔너리 형태로 저장되어 `agent.logs`에 추가됩니다.
- `agent.write_memory_to_messages()`는 에이전트의 메모리를 모델이 볼 수 있는 채팅 메시지 목록으로 변환합니다. 이 메소드는 로그의 각 단계를 살펴보고 중요한 내용만 메시지로 저장합니다. 예를 들어, 시스템 프롬프트와 작업을 각각 별도 메시지로 저장하고, 각 단계의 LLM 출력과 도구 호출 결과를 개별 메시지로 저장합니다. 전체적인 흐름 파악이 필요할 때 권장드립니다. 단, 모든 로그가 이 메소드를 통해 기록되는 것은 아닙니다.

## 도구[[tools]]

도구는 에이전트가 사용할 수 있는 독립적인 함수입니다. LLM이 도구를 사용하기 위해서는 먼저 API를 구성해야하며, 또한 LLM에게 해당 도구 호출하는 방법을 설명해주어야합니다 :
- 이름
- 설명
- 입력 타입과 설명
- 출력 타입

예를 들어 [`PythonInterpreterTool`]을 확인할 수 있습니다: 이름, 설명, 입력 설명, 출력 타입, 그리고 액션을 수행하는 `forward` 메소드가 있습니다.

에이전트가 초기화될 때 도구 속성이 에이전트의 시스템 프롬프트에 포함되는 도구 설명을 생성하는 데 사용됩니다. 이를 통해 에이전트는 사용할 수 있는 도구와 그 이유를 알 수 있습니다.

**스키마 정보**: `output_schema`가 정의된 도구(구조화된 출력을 가진 MCP 도구 등)의 경우, `CodeAgent` 시스템 프롬프트에 자동으로 JSON 스키마 정보가 포함됩니다. 이는 에이전트가 도구 출력의 예상 구조를 이해하고 데이터에 적절히 접근할 수 있도록 도와줍니다.

### 기본 툴박스[[default-toolbox]]

"toolkit" extra와 함께 `smolagents`를 설치하면 에이전트를 강화하는 기본 툴박스가 함께 제공되며, `add_base_tools=True` 인수로 초기화 시 에이전트에 추가할 수 있습니다:

- **DuckDuckGo 웹 검색***: DuckDuckGo 브라우저를 사용하여 웹 검색을 수행합니다.
- **Python 코드 인터프리터**: 보안 환경에서 LLM이 생성한 Python 코드를 실행합니다. 이 도구는 코드 기반 에이전트가 이미 기본적으로 Python 코드를 실행할 수 있으므로 `add_base_tools=True`로 초기화할 때만 [`ToolCallingAgent`]에 추가됩니다.
- **Transcriber**: 오디오를 텍스트로 변환하는 Whisper-Turbo 기반의 음성-텍스트 파이프라인입니다.

인수와 함께 호출하여 도구를 수동으로 사용할 수 있습니다.

```python
# !pip install smolagents[toolkit]
from smolagents import WebSearchTool

search_tool = WebSearchTool()
print(search_tool("Who's the current president of Russia?"))
```

### 새로운 도구 생성[[create-a-new-tool]]

Hugging Face의 기본 도구가 다루지 않는 사용 사례를 위해 자신만의 도구를 만들 수 있습니다.
예를 들어, Hub에서 주어진 작업에 대해 가장 많이 다운로드된 모델을 반환하는 도구를 만들어보겠습니다.

아래 코드부터 시작하겠습니다.

```python
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(most_downloaded_model.id)
```

이 코드는 함수로 만들고 `tool` 데코레이터를 추가하여 간단히 도구로 변환할 수 있습니다.
하지만 이것이 도구를 만드는 유일한 방법은 아닙니다. [Tool]의 하위 클래스로 직접 정의하는 방법도 있으며, 이 방식은 더 많은 유연성을 제공합니다. 예를 들어 리소스 집약적인 클래스 속성을 초기화할 때 유용합니다.

두 옵션 모두에서 어떻게 작동하는지 살펴보겠습니다:

<hfoptions id="build-a-tool">
<hfoption id="Decorate a function with @tool">

```py
from smolagents import tool

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id
```

함수에는 다음이 필요합니다:
- 명확한 이름. 이름은 에이전트를 구동하는 LLM이 이해할 수 있도록 이 도구가 무엇을 하는지 충분히 설명적이어야 합니다. 이 도구는 작업에 대해 가장 많이 다운로드된 모델을 반환하므로 `model_download_tool`이라고 명명하겠습니다.
- 입력과 출력 모두에 대한 타입 힌트
- 각 인수가 설명되는 'Args:' 부분을 포함하는 설명(이번에는 타입 표시 없이, 타입 힌트에서 가져옵니다). 도구 이름과 마찬가지로, 이 설명은 에이전트를 구동하는 LLM을 위한 설명서이므로 소홀히 하지 마세요.

이 모든 요소는 초기화 시 에이전트의 시스템 프롬프트에 자동으로 포함됩니다: 따라서 최대한 명확하게 만들도록 노력하세요!

> [!TIP]
> 이 정의 형식은 `apply_chat_template`에서 사용되는 도구 스키마와 동일하며, 유일한 차이점은 추가된 `tool` 데코레이터입니다: 도구 사용 API에 대해 더 자세히 알아보려면 [여기](https://huggingface.co/blog/unified-tool-use#passing-tools-to-a-chat-template)를 읽어보세요.


그런 다음 에이전트를 직접 초기화할 수 있습니다:
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```
</hfoption>
<hfoption id="Subclass Tool">

```py
from smolagents import Tool

class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return most_downloaded_model.id
```

하위 클래스에는 다음 속성이 필요합니다:
- 명확한 `name` (이름). 에이전트를 구동하는 LLM이 도구의 기능을 이해할 수 있도록 이름에 대해 충분히 설명해야 합니다. 이 도구는 작업에 대해 가장 많이 다운로드된 모델을 반환하므로 `model_download_tool`이라고 명명하겠습니다.
- `description`. `name`과 마찬가지로, 이 설명은 에이전트를 구동하는 LLM을 위한 설명서이므로 소홀히 하지 마세요.
- 입력 타입과 설명
- 출력 타입
이 모든 속성은 초기화 시 에이전트의 시스템 프롬프트에 자동으로 포함됩니다: 따라서 최대한 명확하게 만들도록 노력하세요!


그런 다음 에이전트를 직접 초기화할 수 있습니다:
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[ModelDownloadTool()], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```
</hfoption>
</hfoptions>

다음 로그를 얻습니다:
```text
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                          │
│ Can you give me the name of the model that has the most downloads in the 'text-to-video' │
│ task on the Hugging Face Hub?                                                            │
│                                                                                          │
╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 model_name = model_download_tool(task="text-to-video")                               │
│   2 print(model_name)                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
ByteDance/AnimateDiff-Lightning

Out: None
[Step 0: Duration 0.27 seconds| Input tokens: 2,069 | Output tokens: 60]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer("ByteDance/AnimateDiff-Lightning")                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: ByteDance/AnimateDiff-Lightning
[Step 1: Duration 0.10 seconds| Input tokens: 4,288 | Output tokens: 148]
Out[20]: 'ByteDance/AnimateDiff-Lightning'
```

> [!TIP]
> 도구에 대해 더 자세히 알아보려면 [전용 튜토리얼](./tutorials/tools#what-is-a-tool-and-how-to-build-one)을 읽어보세요.

## 멀티 에이전트[[multi-agents]]

멀티 에이전트 시스템은 Microsoft의 프레임워크 [Autogen](https://huggingface.co/papers/2308.08155)과 함께 도입되었습니다.

이러한 프레임워크에서는 단일 에이전트 대신 여러 에이전트가 협력하여 작업을 해결합니다.
실제로 대부분의 벤치마크에서 더 우수한 성능을 보여줍니다. 성능이 향상되는 이유는 개념적으로 단순합니다. 많은 작업에서 모든 기능을 담당하는 범용 시스템보다는 특정 하위 작업에 특화된 전문 단위를 사용하는 것이 더 효과적이기 때문입니다. 서로 다른 도구와 메모리를 가진 에이전트들을 활용하면 효율적인 역할 분담이 가능합니다. 예를 들어, 웹 검색 에이전트가 수집한 모든 웹페이지 내용을 코드 생성 에이전트의 메모리에까지 저장할 필요가 있을까요? 각자의 역할에 맞게 분리해서 운영하는 것이 훨씬 효율적입니다.

`smolagents`로 계층적 멀티 에이전트 시스템을 쉽게 구축할 수 있습니다.

이를 위해서는 에이전트에 `name`과 `description` 속성만 있으면 되며, 이는 도구와 마찬가지로 관리자 에이전트의 시스템 프롬프트에 포함되어 관리되는 에이전트를 호출하는 방법을 알려줍니다.
그런 다음 관리자 에이전트를 초기화할 때 `managed_agents` 매개변수에 이 관리되는 에이전트를 전달할 수 있습니다.

다음은 네이티브 [`WebSearchTool`]을 사용하여 특정 웹 검색 에이전트를 관리하는 에이전트를 만드는 예시입니다:

```py
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> 효율적인 멀티 에이전트 구현의 심화 예제를 보려면 [멀티 에이전트 시스템을 GAIA 리더보드 상위권으로 끌어올린 방법](https://huggingface.co/blog/beating-gaia)을 확인하세요.

## 에이전트와 대화하고 멋진 Gradio 인터페이스에서 그 사고 과정을 시각화하기[[talk-with-your-agent-and-visualize-its-thoughts-in-a-cool-gradio-interface]]

`GradioUI`를 사용하여 에이전트에 대화형으로 작업을 제출하고 그 사고와 실행 과정을 관찰할 수 있습니다. 다음은 예시입니다:

```py
from smolagents import (
    load_tool,
    CodeAgent,
    InferenceClientModel,
    GradioUI
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = InferenceClientModel(model_id=model_id)

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
```

내부적으로 사용자가 새로운 요청을 입력하면 에이전트는 `agent.run(user_request, reset=False)`로 실행됩니다.
`reset=False` 플래그는 이 새로운 작업을 실행하기 전에 에이전트의 메모리가 플러시되지 않음을 의미하며, 이를 통해 대화가 계속될 수 있습니다.

다른 에이전트 애플리케이션에서도 이 `reset=False` 인수를 사용하여 대화를 계속할 수 있습니다.

Gradio UI에서 사용자가 실행 중인 에이전트를 중단할 수 있도록 하려면 `agent.interrupt()` 메소드를 트리거하는 버튼으로 이를 수행할 수 있습니다.
이렇게 하면 현재 단계가 끝날 때 에이전트가 중지되고 오류가 발생합니다.

## 다음 단계[[next-steps]]

마지막으로 에이전트를 필요에 맞게 구성했다면 Hub에 공유할 수 있습니다!

```py
agent.push_to_hub("m-ric/my_agent")
```

마찬가지로, 도구의 코드를 신뢰한다면 Hub에 업로드된 에이전트를 불러오려면 다음을 사용하세요:
```py
agent.from_hub("m-ric/my_agent", trust_remote_code=True)
```

더 자세한 활용법을 원한다면 다음 튜토리얼들을 참고하세요:
- [코드 에이전트가 작동하는 방법에 대한 설명](./tutorials/secure_code_execution)
- [좋은 에이전트를 구축하는 방법에 대한 가이드](./tutorials/building_good_agents).
- [도구 사용에 대한 상세 가이드](./tutorials/building_good_agents).
