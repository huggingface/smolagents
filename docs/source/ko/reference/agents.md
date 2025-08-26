# 에이전트[[agents]]

<Tip warning={true}>

Smolagents는 언제든지 변경될 수 있는 실험적인 API입니다. 에이전트가 반환하는 결과는
API나 기반 모델이 변경될 수 있으므로 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 자세히 알아보려면 [소개 가이드](../index)를 꼭 읽어보세요. 이 페이지는
기본 클래스에 대한 API 문서를 포함하고 있습니다.

## 에이전트[[agents]]

저희 에이전트는 [`MultiStepAgent`]를 상속받으며, 이는 여러 단계로 행동할 수 있음을 의미합니다. 각 단계는 하나의 생각, 그리고 하나의 도구 호출 및 실행으로 구성됩니다. [이 개념 가이드](../conceptual_guides/react)에서 더 자세히 알아보세요.

저희는 메인 [`Agent`] 클래스를 기반으로 두 가지 유형의 에이전트를 제공합니다.
  - [`CodeAgent`]는 도구 호출을 Python 코드로 작성합니다 (이것이 기본값입니다).
  - [`ToolCallingAgent`]는 도구 호출을 JSON으로 작성합니다.

두 에이전트 모두 초기화 시 `model` 인수와 도구 목록인 `tools`가 필요합니다.

### 에이전트 클래스[[classes-of-agents]]

[[autodoc]] MultiStepAgent

[[autodoc]] CodeAgent

[[autodoc]] ToolCallingAgent

### stream_to_gradio[[stream_to_gradio]]

[[autodoc]] stream_to_gradio

### GradioUI[[gradio-ui]]

> [!TIP]
> UI를 사용하려면 `gradio`가 설치되어 있어야 합니다. 설치되어 있지 않다면 `pip install 'smolagents[gradio]'`를 실행해주세요.

[[autodoc]] GradioUI

## 프롬프트[[prompts]]

[[autodoc]] smolagents.agents.PromptTemplates

[[autodoc]] smolagents.agents.PlanningPromptTemplate

[[autodoc]] smolagents.agents.ManagedAgentPromptTemplate

[[autodoc]] smolagents.agents.FinalAnswerPromptTemplate

## 메모리[[memory]]

Smolagents는 여러 단계에 걸쳐 정보를 저장하기 위해 메모리를 사용합니다.

[[autodoc]] smolagents.memory.AgentMemory

## Python 코드 실행기[[python-code-executors]]

[[autodoc]] smolagents.local_python_executor.PythonExecutor

### 로컬 Python 실행기[[local-python-executor]]

[[autodoc]] smolagents.local_python_executor.LocalPythonExecutor

### 원격 Python 실행기[[remote-python-executors]]

[[autodoc]] smolagents.remote_executors.RemotePythonExecutor

#### E2BExecutor[[e2bexecutor]]

[[autodoc]] smolagents.remote_executors.E2BExecutor

#### DockerExecutor[[dockerexecutor]]

[[autodoc]] smolagents.remote_executors.DockerExecutor

#### WasmExecutor[[wasmexecutor]]

[[autodoc]] smolagents.remote_executors.WasmExecutor