# Human-in-the-Loop: 사용자와 상호작용하며 에이전트 계획 수정하기 [[humanintheloop-customize-agent-plan-interactively]]

이 페이지에서는 smolagents 라이브러리의 고급 사용법을 소개합니다. 특히 사용자와의 상호작용을 통한 계획 생성, 계획 수정, 그리고 에이전트 워크플로에서의 메모리 보존을 위한 Human-in-the-Loop (HITL) 접근 방식을 중점적으로 설명합니다.
예제는 `examples/plan_customization/plan_customization.py`의 코드를 기반으로 합니다.

## 개요 [[overview]]

이 예제는 다음과 같은 Human-in-the-Loop 전략을 구현하는 방법을 안내합니다.

- 단계 콜백(step callback)을 사용하여 계획 생성 후 에이전트 실행 중단하기
- 사용자가 실행 전에 에이전트의 계획을 검토하고 수정할 수 있도록 지원 (Human-in-the-Loop)
- 에이전트의 메모리를 보존하면서 실행 재개하기
- 사용자 피드백을 기반으로 계획을 동적으로 업데이트하여 사용자가 제어권을 유지하도록 지원

## 핵심 개념 [[key-concepts]]

### 단계 콜백을 이용한 계획 중단 [[step-callbacks-for-plan-interruption]]

에이전트가 계획을 생성한 후 일시 중지하도록 설정할 수 있습니다. 이는 PlanningStep에 단계 콜백을 등록하여 구현합니다.

```python
agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[DuckDuckGoSearchTool()],
    planning_interval=5,  # 5단계마다 계획
    step_callbacks={PlanningStep: interrupt_after_plan},
    max_steps=10,
    verbosity_level=1
)
```

### Human-in-the-Loop: 대화형 계획 검토 및 수정 [[humanintheloop-interactive-plan-review-and-modification]]

에이전트가 계획을 생성하면, 콜백 함수가 해당 계획을 사용자에게 보여주고 다음 옵션 중 하나를 선택하도록 안내합니다.

1. 계획 승인
2. 계획 수정
3. 실행 취소

예제 상호작용:

```
============================================================
🤖 에이전트 계획 생성됨
============================================================
1. 최근 AI 발전 사항 검색
2. 상위 결과 분석
3. 가장 중요한 3가지 돌파구 요약
4. 각 돌파구에 대한 소스 포함
============================================================

옵션을 선택하세요.
1. 계획 승인
2. 계획 수정
3. 취소
선택 (1-3):
```

이 Human-in-the-Loop 단계를 통해 사용자는 실행이 계속되기 전에 개입하여 계획을 검토하고 수정할 수 있으며, 이를 통해 에이전트의 행동이 사용자의 의도와 일치하도록 보장할 수 있습니다.

사용자가 수정을 선택하면 계획을 직접 편집할 수 있으며, 업데이트된 계획은 이후 실행 단계에서 사용됩니다.

### 메모리 보존 및 실행 재개 [[memory-preservation-and-resuming-execution]]

`reset=False` 옵션으로 에이전트를 실행하면 이전의 모든 단계와 메모리가 보존됩니다. 이를 통해 중단 또는 계획 수정 후에도 실행을 이어서 진행할 수 있습니다.

```python
# 첫 번째 실행 (중단될 수 있음)
agent.run(task, reset=True)

# 보존된 메모리로 재개
agent.run(task, reset=False)
```

### 에이전트 메모리 검사 [[inspecting-agent-memory]]

에이전트의 메모리를 검사하여 지금까지 수행된 모든 단계를 확인할 수 있습니다.

```python
print(f"현재 메모리에 {len(agent.memory.steps)}개의 단계가 포함되어 있습니다.")
for i, step in enumerate(agent.memory.steps):
    step_type = type(step).__name__
    print(f"  {i+1}. {step_type}")
```

## Human-in-the-Loop 워크플로우 예시 [[example-humanintheloop-workflow]]

1. 에이전트가 복잡한 작업을 받아 실행을 시작합니다.
2. 계획 단계가 생성되고, 사용자 검토를 위해 실행이 일시 중지됩니다.
3. 사용자가 계획을 검토하고 필요에 따라 수정합니다 (Human-in-the-Loop).
4. 승인되거나 수정된 계획으로 실행을 재개합니다.
5. 모든 단계는 향후 실행을 위해 보존되어 투명성과 제어권을 보장합니다.

## 오류 처리 [[error-handling]]

예제는 다음에 대한 오류 처리를 포함합니다.
- 사용자 취소
- 계획 수정 오류
- 실행 재개 실패

## 요구사항 [[requirements]]

- smolagents 라이브러리
- DuckDuckGoSearchTool (smolagents에 포함)
- InferenceClientModel (🤗 Hugging Face API 토큰 필요)

## 교육적 가치 [[educational-value]]

이 예제는 다음을 시연합니다.
- 사용자 정의 에이전트 동작을 위한 단계 콜백 구현 방법
- 다중 단계 에이전트의 메모리 관리 기법
- 에이전트 시스템의 사용자 상호작용 패턴
- 동적 에이전트 제어를 위한 계획 수정 기법
- 대화형 에이전트 시스템의 오류 처리 방법

---

전체 코드는 [`examples/plan_customization`](https://github.com/huggingface/smolagents/tree/main/examples/plan_customization)에서 확인하세요.
