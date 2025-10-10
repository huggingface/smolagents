# 📚 에이전트 메모리 관리[[-manage-your-agents-memory]]

[[open-in-colab]]

결국 에이전트는 도구와 프롬프트로 이루어진 단순한 구성요소로 정의됩니다.
그리고 무엇보다 중요한 것은 에이전트가 과거 단계의 메모리를 가지고 있어 계획, 실행, 오류의 이력을 추적한다는 점입니다.

### 에이전트 메모리 재생[[replay-your-agents-memory]]

과거 실행된 에이전트를 확인하기 위한 몇 가지 기능을 제공합니다.

[계측 가이드](./inspect_runs)에서 언급한 바와 같이, 에이전트 실행을 계측하여 특정 단계를 확대하거나 축소할 수 있는 우수한 UI로 시각화할 수 있습니다.

또한 다음과 같이 `agent.replay()`를 사용할 수도 있습니다.

에이전트를 실행한 후,
```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=0)

result = agent.run("What's the 20th Fibonacci number?")
```

이 마지막 실행을 다시 재생하고 싶다면, 다음 코드를 사용하면 됩니다.
```py
agent.replay()
```

### 에이전트 메모리 동적 변경[[dynamically-change-the-agents-memory]]

많은 고급 사용 사례에서는 에이전트의 메모리를 동적으로 수정해야 합니다.

에이전트의 메모리는 다음과 같이 접근할 수 있습니다.


```py
from smolagents import ActionStep

system_prompt_step = agent.memory.system_prompt
print("The system prompt given to the agent was:")
print(system_prompt_step.system_prompt)

task_step = agent.memory.steps[0]
print("\n\nThe first task step was:")
print(task_step.task)

for step in agent.memory.steps:
    if isinstance(step, ActionStep):
        if step.error is not None:
            print(f"\nStep {step.step_number} got this error:\n{step.error}\n")
        else:
            print(f"\nStep {step.step_number} got these observations:\n{step.observations}\n")
```

`agent.memory.get_full_steps()`를 사용하여 전체 단계를 딕셔너리 형태로 가져올 수 있습니다.

또한 단계 콜백을 사용하여 에이전트의 메모리를 동적으로 변경할 수도 있습니다.

단계 콜백은 인자로 `agent` 객체 자체에 접근할 수 있으므로, 위에서 설명한 것처럼 모든 메모리 단계에 접근하여 필요한 경우 수정할 수 있습니다. 예를 들어, 웹 브라우저 에이전트가 수행하는 각 단계의 스크린샷을 관찰하고 있다고 가정해 보겠습니다. 이 경우 최신 스크린샷은 유지하면서 토큰 비용을 절약하기 위해 이전 단계의 이미지를 메모리에서 제거할 수 있습니다.

이 경우 다음과 같은 코드를 사용할 수 있습니다.
_주의: 이 코드는 간결함을 위해 일부 임포트 및 객체 정의가 생략된 불완전한 예시입니다. 전체 작동 버전의 코드는 [원본 스크립트](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py)에서 확인하세요._

```py
import helium
from PIL import Image
from io import BytesIO
from time import sleep

def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # JavaScript 애니메이션이 완료된 후에 스크린샷을 찍도록 합니다.
    driver = helium.get_driver()
    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:  # 이전 스크린샷을 로그에서 제거하여 처리 과정을 간소화합니다.
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= latest_step - 2:
            previous_memory_step.observations_images = None
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]
```

그 다음 에이전트를 초기화할 때 이 함수를 다음과 같이 `step_callbacks` 인수에 전달해야 합니다.

```py
CodeAgent(
    tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[update_screenshot],
    max_steps=20,
    verbosity_level=2,
)
```

전체 작동 예시는 [비전 웹 브라우저 코드](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py)에서 확인할 수 있습니다.

### 에이전트를 단계별로 실행[[run-agents-one-step-at-a-time]]

이 기능은 도구 호출에 오랜 시간이 걸리는 경우에 유용합니다.
에이전트를 한 단계씩 실행하면서 각 단계에서 메모리를 업데이트할 수 있습니다.

```py
from smolagents import InferenceClientModel, CodeAgent, ActionStep, TaskStep

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=1)
agent.python_executor.send_tools({**agent.tools})
print(agent.memory.system_prompt)

task = "What is the 20th Fibonacci number?"

# 필요에 따라 다른 에이전트의 메모리를 불러와 메모리를 수정할 수 있습니다.
# agent.memory.steps = previous_agent.memory.steps

# 새로운 작업을 시작합니다!
agent.memory.steps.append(TaskStep(task=task, task_images=[]))

final_answer = None
step_number = 1
while final_answer is None and step_number <= 10:
    memory_step = ActionStep(
        step_number=step_number,
        observations_images=[],
    )
    # 한 단계를 실행합니다.
    final_answer = agent.step(memory_step)
    agent.memory.steps.append(memory_step)
    step_number += 1

    # 필요한 경우 메모리를 수정할 수도 있습니다
    # 예를 들어 최신 단계를 업데이트 하려면 다음과 같이 처리합니다:
    # agent.memory.steps[-1] = ...

print("The final answer is:", final_answer)
```
