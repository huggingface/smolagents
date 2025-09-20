# Human-in-the-Loop: 에이전트 계획을 대화형으로 맞춤 설정하다 [[humanintheloop-customize-agent-plan-interactively]]

이 페이지는 smolagents 라이브러리의 고급 사용법을 시연하며, 대화형 계획 생성, 사용자 주도 계획 수정, 그리고 에이전트 워크플로우에서의 메모리 보존을 위한 **Human-in-the-Loop (HITL)** 접근 방식에 특별히 중점을 둡니다.
예제는 `examples/plan_customization/plan_customization.py`의 코드를 기반으로 합니다.

## 개요 [[overview]]

이 예제는 다음을 위한 Human-in-the-Loop 전략을 구현하는 방법을 가르칩니다:

- 계획이 생성된 후 에이전트 실행을 중단 (단계 콜백 사용)
- 사용자가 실행 전에 에이전트의 계획을 검토하고 수정할 수 있도록 허용 (Human-in-the-Loop)
- 에이전트의 메모리를 보존하면서 실행 재개
- 사용자 피드백을 기반으로 계획을 동적으로 업데이트하여 인간이 제어권을 유지

## 핵심 개념 [[key-concepts]]

### 계획 중단을 위한 단계 콜백 [[step-callbacks-for-plan-interruption]]

에이전트는 계획을 생성한 후 일시정지하도록 구성됩니다. 이는 `PlanningStep`에 단계 콜백을 등록함으로써 달성됩니다:

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

에이전트가 계획을 생성하면 콜백이 이를 표시하고 인간 사용자에게 다음을 선택하도록 요청합니다:

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

옵션을 선택하세요:
1. 계획 승인
2. 계획 수정
3. 취소
선택 (1-3):
```

이 Human-in-the-Loop 단계는 인간이 실행이 계속되기 전에 개입하여 계획을 검토하거나 수정할 수 있게 하며, 에이전트의 행동이 인간의 의도와 일치하도록 보장합니다.

사용자가 수정을 선택하면 계획을 직접 편집할 수 있습니다. 업데이트된 계획은 이후 실행 단계에서 사용됩니다.

### 메모리 보존 및 실행 재개 [[memory-preservation-and-resuming-execution]]

`reset=False`로 에이전트를 실행하면 이전의 모든 단계와 메모리가 보존됩니다. 이를 통해 중단이나 계획 수정 후에도 실행을 재개할 수 있습니다:

```python
# 첫 번째 실행 (중단될 수 있음)
agent.run(task, reset=True)

# 보존된 메모리로 재개
agent.run(task, reset=False)
```

### 에이전트 메모리 검사 [[inspecting-agent-memory]]

에이전트의 메모리를 검사하여 지금까지 수행된 모든 단계를 확인할 수 있습니다:

```python
print(f"현재 메모리에 {len(agent.memory.steps)}개의 단계가 포함되어 있습니다:")
for i, step in enumerate(agent.memory.steps):
    step_type = type(step).__name__
    print(f"  {i+1}. {step_type}")
```

## 예제 Human-in-the-Loop 워크플로우 [[example-humanintheloop-workflow]]

1. 에이전트가 복잡한 작업으로 시작
2. 계획 단계가 생성되고 인간 검토를 위해 실행이 일시정지
3. 인간이 계획을 검토하고 선택적으로 수정 (Human-in-the-Loop)
4. 승인/수정된 계획으로 실행 재개
5. 향후 실행을 위해 모든 단계가 보존되어 투명성과 제어권 유지

## 오류 처리 [[error-handling]]

예제는 다음에 대한 오류 처리를 포함합니다:
- 사용자 취소
- 계획 수정 오류
- 실행 재개 실패

## 요구사항 [[requirements]]

- smolagents 라이브러리
- DuckDuckGoSearchTool (smolagents에 포함)
- InferenceClientModel (🤗 Hugging Face API 토큰 필요)

## 교육적 가치 [[educational-value]]

이 예제는 다음을 시연합니다:
- 사용자 정의 에이전트 동작을 위한 단계 콜백 구현
- 다단계 에이전트에서의 메모리 관리
- 에이전트 시스템에서의 사용자 상호작용 패턴
- 동적 에이전트 제어를 위한 계획 수정 기법
- 대화형 에이전트 시스템에서의 오류 처리

---

전체 코드는 [`examples/plan_customization`](https://github.com/huggingface/smolagents/tree/main/examples/plan_customization)에서 확인하세요.