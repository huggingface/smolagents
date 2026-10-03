# 도구[[tools]]

<Tip warning={true}>

Smolagents는 언제든지 변경될 수 있는 실험적인 API입니다. API나 사용되는 모델이 변경될 수 있기 때문에 에이전트가 반환하는 결과도 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 자세히 알아보려면 [소개 가이드](../index)를 꼭 읽어보세요. 이 페이지에는 기본 클래스에 대한 API 문서가 포함되어 있습니다.

## 도구 기본 클래스[[tool-base-classes]]

### load_tool[[smolagents.load_tool]]

[[autodoc]] load_tool

### tool[[smolagents.tool]]

[[autodoc]] tool

### Tool[[smolagents.Tool]]

[[autodoc]] Tool

### launch_gradio_demo[[smolagents.launch_gradio_demo]]

[[autodoc]] launch_gradio_demo

## ToolCollection[[smolagents.ToolCollection]]

[[autodoc]] ToolCollection

## MCP 클라이언트[[smolagents.MCPClient]]

[[autodoc]] smolagents.mcp_client.MCPClient

## 에이전트 타입[[agent-types]]

에이전트는 도구 간에 모든 유형의 객체를 처리할 수 있습니다. 각 도구는 완전한 멀티모달을 지원하므로 텍스트, 이미지, 오디오, 비디오 등 다양한 형태의 데이터를 입력받거나 반환할 수 있습니다. 도구 간의 호환성을 높이고, ipython(jupyter, colab, ipython 노트북 등)에서 이러한 반환값을 올바르게 렌더링되록 하기 위해 래퍼 클래스를 구현하여 이러한 타입들을 감쌉니다.

래퍼 객체는 본래의 동작을 유지해야 합니다. 예를 들어, 텍스트 객체는 여전히 문자열처럼 동작해야 하고, 이미지 객체는 여전히 `PIL.Image`처럼 동작해야 합니다.

이러한 타입들은 세 가지 특정 목적을 가집니다:

- 타입에 `to_raw`를 호출하면 기본 객체를 반환해야 합니다.
- 타입에 `to_string`을 호출하면 객체를 문자열로 반환해야 합니다. `AgentText`의 경우에는 해당 문자열이 될 수 있지만, 그 외의 다른 인스턴스에서는 객체의 직렬화된 버전의 경로가 반환됩니다.
- ipython 커널에 표시할 때 객체가 올바르게 표시되어야 합니다.

### AgentText[[smolagents.AgentText]]

[[autodoc]] smolagents.agent_types.AgentText

### AgentImage[[smolagents.AgentImage]]

[[autodoc]] smolagents.agent_types.AgentImage

### AgentAudio[[smolagents.AgentAudio]]

[[autodoc]] smolagents.agent_types.AgentAudio