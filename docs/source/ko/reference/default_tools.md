# 내장 도구[[built-in-tools]]

내장 도구는 `smolagents` 라이브러리에서 제공하는 바로 사용할 수 있는 도구 구현체입니다.

이 내장 도구들은 [`Tool`] 기본 클래스의 구현체로, 각각 웹 검색, Python 코드 실행, 웹 페이지 검색, 사용자 상호작용과 같은 특정 작업을 위해 설계되었습니다.
이 도구들을 사용하면 기본 기능을 직접 구현할 필요 없이 에이전트에서 바로 사용할 수 있습니다.
각 도구는 특정 기능을 처리하고 일관된 인터페이스를 따르므로, 강력한 에이전트 워크플로우로 쉽게 구성할 수 있습니다.

내장 도구는 주요 기능에 따라 다음과 같이 분류할 수 있습니다.
- **정보 검색**: 웹 및 특정 지식 소스에서 정보를 검색하고 가져옵니다.
  - [`ApiWebSearchTool`]
  - [`DuckDuckGoSearchTool`]
  - [`GoogleSearchTool`]
  - [`WebSearchTool`]
  - [`WikipediaSearchTool`]
- **웹 상호작용**: 특정 웹 페이지에서 콘텐츠를 가져와 처리합니다.
  - [`VisitWebpageTool`]
- **코드 실행**: 계산 작업을 위한 Python 코드의 동적 실행합니다.
  - [`PythonInterpreterTool`]
- **사용자 상호작용**: 에이전트와 사용자 간의 Human-in-the-Loop 협업을 활성화합니다.
  - [`UserInputTool`]: 사용자로부터 입력을 수집합니다.
- **음성 처리**: 오디오를 텍스트 데이터로 변환합니다.
  - [`SpeechToTextTool`]
- **워크플로우 제어**: 에이전트 작업의 흐름을 관리하고 지시합니다.
  - [`FinalAnswerTool`]: 최종 응답으로 에이전트 워크플로우를 마무리합니다.

## ApiWebSearchTool[[smolagents.ApiWebSearchTool]]

[[autodoc]] smolagents.default_tools.ApiWebSearchTool

## DuckDuckGoSearchTool[[smolagents.DuckDuckGoSearchTool]]

[[autodoc]] smolagents.default_tools.DuckDuckGoSearchTool

## FinalAnswerTool[[smolagents.FinalAnswerTool]]

[[autodoc]] smolagents.default_tools.FinalAnswerTool

## GoogleSearchTool[[smolagents.GoogleSearchTool]]

[[autodoc]] smolagents.default_tools.GoogleSearchTool

## PythonInterpreterTool[[smolagents.PythonInterpreterTool]]

[[autodoc]] smolagents.default_tools.PythonInterpreterTool

## SpeechToTextTool[[smolagents.SpeechToTextTool]]

[[autodoc]] smolagents.default_tools.SpeechToTextTool

## UserInputTool[[smolagents.UserInputTool]]

[[autodoc]] smolagents.default_tools.UserInputTool

## VisitWebpageTool[[smolagents.VisitWebpageTool]]

[[autodoc]] smolagents.default_tools.VisitWebpageTool

## WebSearchTool[[smolagents.WebSearchTool]]

[[autodoc]] smolagents.default_tools.WebSearchTool

## WikipediaSearchTool[[smolagents.WikipediaSearchTool]]

[[autodoc]] smolagents.default_tools.WikipediaSearchTool