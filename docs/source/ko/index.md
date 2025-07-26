# `smolagents`[[smolagents]]

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png" style="max-width:700px"/>
</div>

## smolagents란 무엇인가요?[[what-is-smolagents]]

`smolagents`는 단 몇 줄의 코드만으로 에이전트를 구축하고 실행할 수 있도록 설계된 오픈소스 Python 라이브러리입니다.

`smolagents`의 주요 특징:

✨ **단순함**: 에이전트 로직이 약 천 줄의 코드로 구현되어 있습니다. 코드 위에 불필요한 복잡한 구조를 추가하지 않고 단순하게 만들었습니다!

🧑‍💻 **코드 에이전트의 완전한 지원**: [`CodeAgent`](reference/agents#smolagents.CodeAgent)는 도구 호출이나 계산 수행을 위해 직접 코드를 작성합니다 ("코드 작성용 에이전트"와는 반대 개념). 이를 통해 함수 중첩, 루프, 조건문 등을 자연스럽게 조합할 수 있습니다. 보안을 위해 [E2B](https://e2b.dev/)나 Docker를 통한 [샌드박스 환경 실행](tutorials/secure_code_execution)을 지원합니다.

📡 **기본 도구 호출 에이전트 지원**: CodeAgent 외에도 [`ToolCallingAgent`](reference/agents#smolagents.ToolCallingAgent)는 일반적인 JSON/텍스트 기반 도구 호출 방식이 필요한 경우를 위해 지원됩니다.

🤗 **Hub 통합**: Gradio Spaces로 에이전트와 도구를 Hub에서 원활하게 공유하고 로드할 수 있습니다.

🌐 **모델 독립적**: Hub의 [Inference providers](https://huggingface.co/docs/inference-providers/index)나 OpenAI, Anthropic 등의 API를 통해 접근하거나, LiteLLM 통합으로 다양한 LLM을 쉽게 연결할 수 있습니다. Transformers나 Ollama를 사용한 로컬 실행도 가능합니다. 원하는 LLM으로 에이전트를 구동하는 것이 간단하고 유연합니다.

👁️ **모달리티 독립적**: 텍스트뿐만 아니라 비전, 비디오, 오디오 입력도 처리할 수 있어 활용 가능한 애플리케이션 범위가 확장됩니다. 비전 관련 [튜토리얼](examples/web_browser)을 확인해보세요.

🛠️ **도구 독립적**: [MCP 서버](reference/tools#smolagents.ToolCollection.from_mcp)의 도구나 [LangChain](reference/tools#smolagents.Tool.from_langchain)의 도구를 사용할 수 있고, [Hub Space](reference/tools#smolagents.Tool.from_space)도 도구로 활용할 수 있습니다.

💻 **CLI 도구**: 보일러플레이트 코드 작성 없이 에이전트를 빠르게 실행할 수 있는 명령줄 유틸리티(smolagent, webagent)가 포함되어 있습니다.

## 빠른 시작[[quickstart]]

[[open-in-colab]]

smolagents를 단 몇 분 만에 시작해보세요! 이 가이드는 첫 번째 에이전트를 생성하고 실행하는 방법을 보여줍니다.

### 설치[[installation]]

pip으로 smolagents를 설치하세요:

```bash
pip install smolagents[toolkit]  # 웹 검색과 같은 기본 도구 포함
```

### 첫 에이전트 만들기[[create-your-first-agent]]

다음은 에이전트를 생성하고 실행하는 최소한의 예제입니다:

```python
from smolagents import CodeAgent, InferenceClientModel

# 모델 초기화 (Hugging Face Inference API 사용)
model = InferenceClientModel()  # 기본 모델 사용

# 도구 없이 에이전트 생성
agent = CodeAgent(tools=[], model=model)

# 작업으로 에이전트 실행
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)
```

끝입니다! 에이전트가 Python 코드를 사용하여 작업을 해결하고 결과를 반환합니다.

### 도구 추가[[adding-tools]]

몇 가지 도구를 추가하여 에이전트를 더 강력하게 만들어보겠습니다:

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

# 이제 에이전트가 웹을 검색할 수 있습니다!
result = agent.run("What is the current weather in Paris?")
print(result)
```

### 다른 모델 사용하기[[using-different-models]]

에이전트와 함께 다양한 모델을 사용할 수 있습니다:

```python
# Hugging Face의 특정 모델 사용
model = InferenceClientModel(model_id="meta-llama/Llama-2-70b-chat-hf")

# OpenAI/Anthropic 사용 (smolagents[litellm] 필요)
from smolagents import LiteLLMModel
model = LiteLLMModel(model_id="gpt-4")

# 로컬 모델 사용 (smolagents[transformers] 필요)
from smolagents import TransformersModel
model = TransformersModel(model_id="meta-llama/Llama-2-7b-chat-hf")
```

## 다음 단계[[next-steps]]

- [설치 가이드](installation)에서 다양한 모델과 도구로 smolagents를 설정하는 방법을 알아보세요
- 더 고급 기능은 [안내서](guided_tour)를 확인하세요
- [커스텀 도구 구축](tutorials/tools)에 대해 알아보세요
- [안전한 코드 실행](tutorials/secure_code_execution)을 살펴보세요
- [멀티 에이전트 시스템](tutorials/building_good_agents) 생성 방법을 확인하세요

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">안내서</div>
      <p class="text-gray-700">기본 사항을 배우고 에이전트 사용에 익숙해지세요. 에이전트를 처음 사용하는 경우 여기서 시작하세요!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">실습 가이드</div>
      <p class="text-gray-700">특정 목표를 달성하는 데 도움이 되는 실용적인 가이드: SQL 쿼리를 생성하고 테스트하는 에이전트를 만들어보세요!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">개념 가이드</div>
      <p class="text-gray-700">중요한 주제에 대한 전체적인 이해를 돕는 설명입니다.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">튜토리얼</div>
      <p class="text-gray-700">에이전트 구축의 중요한 측면을 다루는 포괄적인 튜토리얼입니다.</p>
    </a>
  </div>
</div>
