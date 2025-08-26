# 에이전트[[agents]]

<Tip warning={true}>

Smolagents는 실험적인 API로 언제든지 변경될 수 있습니다. API나 사용되는 모델이 변경될 수 있기 때문에 에이전트가 반환하는 결과도 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 자세히 알아보려면 [소개 가이드](../index)를 꼭 읽어보세요. 이 페이지에는 기본 클래스에 대한 API 문서가 포함되어 있습니다.

## 에이전트[[agents]]

저희 에이전트는 [`MultiStepAgent`]를 상속받으며, 이는 하나의 생각과 하나의 도구 호출 및 실행으로 구성된 여러 단계를 수행할 수 있음을 의미합니다. 이 [개념 가이드](../conceptual_guides/react)에서 더 자세히 알아보세요.

저희는 메인 [`Agent`] 클래스를 기반으로 두 가지 유형의 에이전트를 제공합니다.
  - [`CodeAgent`]는 Python 코드로 도구 호출을 작성합니다.(이것이 기본값입니다.)
  - [`ToolCallingAgent`]는 JSON 형식으로 도구 호출을 작성합니다.

두 경우 모두 초기화 시 `model`과 도구 목록인 `tools`를 인수로 요구합니다.

### 에이전트 클래스[[smolagents.MultiStepAgent]]

[[autodoc]] MultiStepAgent

[[autodoc]] CodeAgent

[[autodoc]] ToolCallingAgent

### stream_to_gradio[[smolagents.stream_to_gradio]]

[[autodoc]] stream_to_gradio

### GradioUI[[smolagents.GradioUI]]

> [!TIP]
> UI를 사용하려면 `gradio`가 설치되어 있어야 합니다. 설치되어 있지 않다면 `pip install 'smolagents[gradio]'`를 실행해주세요.

[[autodoc]] GradioUI

## 프롬프트[[smolagents.PromptTemplates]]

[[autodoc]] smolagents.agents.PromptTemplates

[[autodoc]] smolagents.agents.PlanningPromptTemplate

[[autodoc]] smolagents.agents.ManagedAgentPromptTemplate

[[autodoc]] smolagents.agents.FinalAnswerPromptTemplate

## 메모리[[smolagents.AgentMemory]]

Smolagents는 여러 단계에 걸쳐 정보를 저장하기 위해 메모리를 사용합니다.

[[autodoc]] smolagents.memory.AgentMemory

## Python 코드 실행기[[smolagents.PythonExecutor]]

[[autodoc]] smolagents.local_python_executor.PythonExecutor

### 로컬 Python 실행기[[smolagents.LocalPythonExecutor]]

[[autodoc]] smolagents.local_python_executor.LocalPythonExecutor

### 원격 Python 실행기[[smolagents.remote_executors.RemotePythonExecutor]]

[[autodoc]] smolagents.remote_executors.RemotePythonExecutor

#### E2BExecutor[[smolagents.E2BExecutor]]

[[autodoc]] smolagents.remote_executors.E2BExecutor

#### DockerExecutor[[smolagents.DockerExecutor]]

[[autodoc]] smolagents.remote_executors.DockerExecutor

#### WasmExecutor[[smolagents.WasmExecutor]]

[[autodoc]] smolagents.remote_executors.WasmExecutor