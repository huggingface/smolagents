# 멀티스텝 에이전트는 어떻게 동작하나요?[[how-do-multi-step-agents-work]]

ReAct 프레임워크([Yao et al., 2022](https://huggingface.co/papers/2210.03629))는 현재 에이전트 구축하는 가장 일반적인 접근 방식입니다.

ReAct라는 이름은 "추론(Reason)"과 "행동(Act)"을 결합한 것입니다. 실제로 이 구조를 따르는 에이전트는 주어진 작업을 해결하기 위해 필요한 만큼 여러 단계를 거칩니다. 각 단계는 추론 단계와 행동 단계로 이루어져 있으며, 행동 단계에서는 작업 해결에 가까워지도록 다양한 도구를 호출합니다.

`smolagents`가 제공하는 모든 에이전트는 ReAct 프레임워크를 추상화한 단일 `MultiStepAgent` 클래스를 기반으로 합니다.

이 클래스는 기본적으로 아래와 같은 루프로 동작하며, 기존 변수와 지식도 에이전트 로그에 함께 반영됩니다.

초기화: 시스템 프롬프트는 `SystemPromptStep`에 저장되고, 사용자가 입력한 쿼리는 `TaskStep`에 기록됩니다.

While 루프 (ReAct 루프):

- `agent.write_memory_to_messages()`를 사용하여 에이전트 로그를 LLM이 읽을 수 있는 [채팅 메시지](https://huggingface.co/docs/transformers/en/chat_templating) 목록에 작성합니다.
- 이 메시지를 `Model` 객체에 전송하여 응답을 받습니다. 에이전트는 응답을 파싱하여 액션(`ToolCallingAgent`의 경우 JSON blob, `CodeAgent`의 경우 코드 스니펫)을 추출합니다.
- 액션을 실행하고 결과를 메모리에 기록합니다(`ActionStep`).
- 각 단계가 끝날 때마다 `agent.step_callbacks`에 정의된 모든 콜백 함수를 실행합니다.

**계획(planning)**이 활성화된 경우에는 주기적으로 계획을 수정하고 이를 `PlanningStep`에 저장할 수 있습니다. 이 과정에는 현재 작업과 관련된 사실을 메모리에 기록하는 것도 포함됩니다.

`CodeAgent`의 경우, 이 과정은 아래 그림과 같이 나타납니다.

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png"
    />
</div>

비디오를 통해 동작 과정을 확인해보세요.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

`smolagents` 라이브러리는 두 가지 버전의 에이전트를 제공합니다.
- 도구 호출을 Python 코드 스니펫 형태로 생성하는 [`CodeAgent`]
- 많은 프레임워크에서 일반적으로 사용하는 방식처럼 도구 호출을 JSON 형태로 작성하는 [`ToolCallingAgent`]

필요에 따라 두 방식 중 어느 것이든 사용할 수 있습니다. 예를 들어, 웹 브라우징처럼 각 페이지 상호작용 후 대기 시간이 필요한 경우 JSON 기반 도구 호출이 잘 맞을 수 있습니다.

> [!TIP]
> 멀티스텝 에이전트에 대해 더 알고 싶다면 허깅페이스 블로그의 [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) 포스트를 읽어보세요.