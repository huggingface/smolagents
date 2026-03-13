# 안전한 코드 실행[[secure-code-execution]]

[[open-in-colab]]

> [!TIP]
> 에이전트 빌드에 익숙하지 않으시다면, 먼저 [에이전트 소개](../conceptual_guides/intro_agents)와 [smolagents 가이드 투어](../guided_tour)를 읽어보시기 바랍니다.

### 코드 에이전트[[code-agents]]

[여러](https://huggingface.co/papers/2402.01030) [연구](https://huggingface.co/papers/2411.01747) [논문](https://huggingface.co/papers/2401.00812)에 따르면, LLM이 자신의 행동(도구 호출)을 코드로 작성하게 하는 것이 현재의 표준 도구 호출 형식보다 훨씬 더 효과적이라고 합니다. 현재 업계 표준은 ‘도구 이름과 사용할 인수를 JSON 형태로 기술하는’ 다양한 변형 방식을 따르고 있습니다.

왜 코드가 더 나을까요? 우리는 컴퓨터가 수행하는 작업을 잘 표현하기 위해 특별히 코드 언어를 만들었기 때문입니다. 만약 JSON 스니펫이 더 나은 방법이었다면, 이 패키지는 JSON으로 작성됐을 것이고, 아마 악마가 우리를 비웃었을 겁니다.

코드는 컴퓨터에서의 작업을 표현하는 더 나은 방법일 뿐입니다. 다음과 같은 장점이 있습니다.
- **구성 가능성:** 파이썬 함수를 정의하는 것처럼 JSON 작업을 중첩하거나 나중에 재사용할 JSON 작업 세트를 정의할 수 있을까요?
- **객체 관리:** `generate_image`와 같은 작업의 출력을 JSON에 어떻게 저장할 수 있을까요?
- **일반성:** 코드는 컴퓨터가 할 수 있는 모든 것을 간단하게 표현하도록 만들어졌습니다.
- **LLM 훈련 코퍼스에서의 표현:** LLM 훈련 코퍼스에 이미 수많은 양질의 작업이 포함되어 있다는 하늘의 축복을 활용하지 않을 이유가 있을까요?

이는 [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030)에서 가져온 아래 그림에 잘 나타나 있습니다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

이것이 우리가 코드 에이전트(이 경우에는 파이썬 에이전트)를 제안하는 데 중점을 두는 이유입니다. 이는 곧 안전한 파이썬 인터프리터를 구축하기 위해 더 많은 노력이 필요함을 의미했습니다.

### 로컬 코드 실행??[[local-code-execution]]

기본적으로 `CodeAgent`는 LLM이 생성한 코드를 사용자의 환경에서 실행합니다.

이는 본질적으로 위험하며, LLM이 생성한 코드는 사용자의 환경을 손상시킬 수 있습니다.

악성 코드 실행은 여러 가지 방식으로 발생할 수 있습니다.
- **단순 LLM 오류:** LLM은 아직 완벽하지 않으며 도움을 주려는 과정에서 의도치 않게 해로운 명령어를 생성할 수 있습니다. 이러한 위험은 낮지만 LLM이 잠재적으로 위험한 코드를 실행하려 시도한 사례가 실제로 관찰된 바 있습니다.
- **공급망 공격:** 신뢰할 수 없거나 손상된 LLM을 실행하면 시스템이 유해한 코드 생성에 노출될 수 있습니다. 안전한 추론 인프라에서 잘 알려진 모델을 사용할 때 이 위험은 매우 낮지만, 이론적인 가능성은 남아 있습니다.
- **프롬프트 인젝션:** 웹을 탐색하는 에이전트가 유해한 지침이 포함된 악성 웹사이트에 도달해 에이전트의 메모리에 공격을 주입할 수 있습니다.
- **공개적으로 접근 가능한 에이전트 악용:** 대중에게 노출된 에이전트는 악의적인 행위자에 의해 악용되어 유해한 코드를 실행할 수 있습니다. 공격자는 에이전트의 실행 능력을 악용하기 위해 적대적인 입력을 만들어 의도하지 않은 결과를 초래할 수 있습니다.
악성 코드가 우발적이든 의도적이든 실행되면 파일 시스템이 손상되고, 로컬 또는 클라우드 리소스가 악용되며, API 서비스가 남용되고, 심지어 네트워크 보안까지 위협받을 수 있습니다.

[에이전시 스펙트럼](../conceptual_guides/intro_agents)에서 코드 에이전트는 다른 덜 에이전트적인 설정에 비해 시스템 내에서 LLM에게 훨씬 더 높은 수준의 에이전시를 부여한다고 할 수 있습니다. 이는 곧 더 높은 수준의 위험을 수반합니다.

따라서 보안에 각별한 주의가 필요합니다.

더 강력한 보안을 위해, 설정 비용은 다소 높지만 보다 높은 수준의 보호를 제공하는 다양한 조치를 제안합니다.

어떤 해결책도 100% 안전하지 않다는 점을 명심하시기 바랍니다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/code_execution_safety_diagram.png">

### 로컬 파이썬 실행기[[our-local-python-executor]]

첫 번째 보안 계층을 추가하기 위해, 일반 파이썬 인터프리터에서 `smolagents`의 코드 실행가 실행되지 않습니다.
우리는 더 안전한 `LocalPythonExecutor`를 처음부터 다시 구축했습니다.

정확히 말하면, 이 인터프리터는 코드에서 추상 구문 트리(AST)를 가져와서 이를 연산 단위로 실행하며 항상 특정 규칙을 따르도록 합니다
- 기본적으로, 사용자가 승인 목록에 명시적으로 추가하지 않는 한 import는 허용되지 않습니다.
- 또한, 하위 모듈에 대한 접근은 기본적으로 비활성화되어 있으며, 각 하위 모듈도 import 목록에서 명시적으로 승인되어야 합니다. 또는 예를 들어 `numpy.*`를 전달하여 `numpy`와 `numpy.random` 또는 `numpy.a.b`와 같은 모든 하위 패키지를 허용할 수 있습니다.
   - `random`과 같이 겉보기에 무해해 보이는 일부 패키지도 `random._os`에서처럼 잠재적으로 유해한 하위 모듈에 대한 접근을 허용할 수 있다는 점에 유의하세요.
- 처리되는 기본 연산의 총 횟수는 무한 루프와 리소스 팽창을 방지하기 위해 제한됩니다.
- 사용자 정의 인터프리터에 명시적으로 정의되지 않은 모든 연산은 오류를 발생시킵니다.

다음과 같이 이러한 안전장치를 시험해 볼 수 있습니다.

```py
from smolagents.local_python_executor import LocalPythonExecutor

# 사용자 정의 실행기 설정, "numpy" 패키지 승인
custom_executor = LocalPythonExecutor(["numpy"])

# 오류 메시지를 가독성이 좋게 출력하는 유틸리티
def run_capture_exception(command: str):
    try:
        custom_executor(harmful_command)
    except Exception as e:
        print("ERROR:\n", e)

# 정의되지 않은 명령어는 작동하지 않음
harmful_command="!echo Bad command"
run_capture_exception(harmful_command)
# >>> ERROR: invalid syntax (<unknown>, line 1)


# os와 같은 import는 `additional_authorized_imports`에 명시적으로 추가되지 않는 한 수행되지 않음
harmful_command="import os; exit_code = os.system('echo Bad command')"
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'import os' due to: InterpreterError: Import of os is not allowed. Authorized imports are: ['statistics', 'numpy', 'itertools', 'time', 'queue', 'collections', 'math', 'random', 're', 'datetime', 'stat', 'unicodedata']

# 승인된 import에서도 잠재적으로 유해한 패키지는 import되지 않음
harmful_command="import random; random._os.system('echo Bad command')"
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'random._os.system('echo Bad command')' due to: InterpreterError: Forbidden access to module: os

# 무한 루프는 N번의 연산 후 중단됨
harmful_command="""
while True:
    pass
"""
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'while True: pass' due to: InterpreterError: Maximum number of 1000000 iterations in While loop exceeded
```

이러한 안전장치 덕분에 인터프리터는 더 안전하게 실행할 수 있습니다.
다양한 사용 사례에서 인터프리터를 사용했지만 사용자 환경에 어떠한 손상도 관찰되지 않았습니다.

> [!WARNING]
> 로컬 파이썬 샌드박스 환경은 본질적으로 완전한 안전성을 보장할 수 없다는 점을 인식하는 것이 중요합니다. 저희 인터프리터는 표준 파이썬 인터프리터보다 안전성이 크게 향상되었지만, 의지가 확고한 공격자나 악의적으로 미세 조정된 LLM이 취약점을 찾아 사용자 환경에 피해를 줄 가능성은 여전히 존재합니다.
> 
> 예를 들어, `Pillow`와 같은 패키지가 이미지를 처리하도록 허용한 경우, LLM은 하드 드라이브를 가득 채울 수천 개의 대용량 이미지 파일을 생성하는 코드를 만들 수 있습니다. 다른 고급 탈출 기술은 승인된 패키지의 더 깊은 취약점을 악용할 수 있습니다.
> 
> LLM이 생성한 코드를 로컬 환경에서 실행하는 것은 항상 위험을 내포합니다. 진정으로 강력한 보안 격리를 통해 LLM 생성 코드를 실행하는 유일한 방법은 아래에 자세히 설명된 E2B나 Docker와 같은 원격 실행 옵션을 사용하는 것입니다.

신뢰할 수 있는 추론 제공업체의 잘 알려진 LLM을 사용할 때 악의적인 공격의 위험은 낮지만 0은 아닙니다.
보안 수준이 높은 애플리케이션이나 신뢰도가 낮은 모델을 사용하는 경우, 원격 실행 샌드박스 사용을 고려해야 합니다.

## 안전한 코드 실행을 위한 샌드박스 접근 방식[[sandbox-approaches-for-secure-code-execution]]

코드를 실행하는 AI 에이전트로 작업할 때 보안은 가장 중요합니다. smolagents에서 코드 실행을 샌드박싱하는 데에는 두 가지 주요 접근 방식이 있으며, 각각 다른 보안 속성과 기능을 가집니다:


![샌드박스 접근 방식 비교](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/sandboxed_execution.png)

1. **개별 코드 스니펫을 샌드박스에서 실행**: 이 접근 방식(다이어그램 왼쪽)은 에이전트가 생성한 파이썬 코드 스니펫만 샌드박스에서 실행하고 나머지 에이전트 시스템은 로컬 환경에 유지합니다. `executor_type="e2b"`, `executor_type="modal"` 또는
`executor_type="docker"`를 사용하여 설정하기가 더 간단하지만, 다중 에이전트를 지원하지 않으며 여전히 환경과 샌드박스 간에 상태 데이터를 전달해야 합니다.

2. **전체 에이전트 시스템을 샌드박스에서 실행**: 이 접근 방식(다이어그램 오른쪽)은 에이전트, 모델, 도구를 포함한 전체 에이전트 시스템을 샌드박스 환경 내에서 실행합니다. 이는 더 나은 격리를 제공하지만 더 많은 수동 설정이 필요하며, 민감한 자격 증명(API 키 등)을 샌드박스 환경으로 전달해야 할 수 있습니다.

이 가이드에서는 에이전트 애플리케이션을 위해 두 가지 유형의 샌드박스 접근 방식을 설정하고 사용하는 방법을 설명합니다.

### E2B 설정[[e2b-setup]]

#### 설치[[installation]]

1. [e2b.dev](https://e2b.dev)에서 E2B 계정을 생성합니다.
2. 필요한 패키지를 설치합니다:
```bash
pip install 'smolagents[e2b]'
```

#### E2B에서 에이전트 실행하기: 빠른 시작[[running-your-agent-in-e2b-quick-start]]

E2B 샌드박스를 사용하는 간단한 방법을 제공합니다. 다음과 같이 에이전트 초기화에 `executor_type="e2b"`를 추가하기만 하면 됩니다:

```py
from smolagents import InferenceClientModel, CodeAgent

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="e2b") as agent:
    agent.run("100번째 피보나치 수를 알려줄 수 있나요?")
```

> [!TIP]
> 에이전트를 컨텍스트 관리자(`with` 문)로 사용하면 에이전트가 작업을 완료한 직후 E2B 샌드박스가 자원 해제 및 정리가 자동으로 수행되도록 보장합니다.
> 또는 에이전트의 `cleanup()` 메소드를 수동으로 호출할 수도 있습니다.

이 솔루션은 각 `agent.run()` 시작 시 에이전트 상태를 서버로 보냅니다.
그런 다음 로컬 환경에서 모델이 호출되지만, 생성된 코드는 실행을 위해 샌드박스로 전송되고 출력만 반환됩니다.

이는 아래 그림에 설명되어 있습니다.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/sandboxed_execution.png" alt="sandboxed code execution" width=60% max-width=500px>
</p>

하지만 [관리형 에이전트](../examples/multiagents)에 대한 모든 호출은 모델 호출을 필요로 하는데, 비밀 정보를 원격 샌드박스로 전송하지 않기 때문에 모델 호출에 자격 증명이 부족하게 됩니다.
따라서 이 솔루션은 더 복잡한 다중 에이전트 설정에서는 아직 작동하지 않습니다.

#### E2B에서 에이전트 실행하기: 다중 에이전트[[running-your-agent-in-e2b-multi-agents]]

E2B 샌드박스에서 다중 에이전트를 사용하려면 E2B 내에서 에이전트를 완전히 실행해야 합니다.

방법은 다음과 같습니다.

```python
from e2b_code_interpreter import Sandbox
import os

# 샌드박스 생성
sandbox = Sandbox()

# 필요한 패키지 설치
sandbox.commands.run("pip install smolagents")

def run_code_raise_errors(sandbox, code: str, verbose: bool = False) -> str:
    execution = sandbox.run_code(
        code,
        envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])

# 에이전트 애플리케이션 정의
agent_code = """
import os
from smolagents import CodeAgent, InferenceClientModel

# 에이전트 초기화
agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[],
    name="coder_agent",
    description="이 에이전트는 코드를 사용하여 어려운 알고리즘 문제를 처리합니다."
)

manager_agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[],
    managed_agents=[agent],
)

# 에이전트 실행
response = manager_agent.run("20번째 피보나치 수는 무엇인가요?")
print(response)
"""

# 샌드박스에서 에이전트 코드 실행
execution_logs = run_code_raise_errors(sandbox, agent_code)
print(execution_logs)
```

### Modal 설정[[modal-setup]]

#### 설치[[installation]]

1. [modal.com](https://modal.com/signup)에서 Modal 계정을 생성합니다.
2. 필요한 패키지를 설치합니다:
```bash
pip install 'smolagents[modal]'
```

#### Modal에서 에이전트 실행하기: 빠른 시작[[running-your-agent-in-modal-quick-start]]

Modal 샌드박스를 사용하는 간단한 방법을 제공합니다. 다음과 같이 에이전트 초기화에 `executor_type="modal"`을 추가하기만 하면 됩니다:

```py
from smolagents import InferenceClientModel, CodeAgent

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="modal") as agent:
    agent.run("42번째 피보나치 수는 무엇인가요?")
```

> [!TIP]
> 에이전트를 컨텍스트 관리자(`with` 문)로 사용하면 에이전트가 작업을 완료한 직후 Modal 샌드박스가 정리되도록 보장합니다.
> 또는 에이전트의 `cleanup()` 메소드를 수동으로 호출할 수도 있습니다.

`InferenceClientModel`에서 생성된 에이전트 상태와 코드는 Modal 샌드박스로 전송되어 안전하게 코드를 실행할 수 있습니다.

### Docker 설정[[docker-setup]]

#### 설치[[installation]]

1. [시스템에 Docker 설치하기](https://docs.docker.com/get-started/get-docker/)
2. 필요한 패키지를 설치합니다.
```bash
pip install 'smolagents[docker]'
```

#### Docker에서 에이전트 실행하기: 빠른 시작[[running-your-agent-in-docker-quick-start]]

위의 E2B 샌드박스와 유사하게, Docker를 빠르게 시작하려면 에이전트 초기화에 `executor_type="docker"`를 추가하기만 하면 됩니다.

```py
from smolagents import InferenceClientModel, CodeAgent

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="docker") as agent:
    agent.run("100번째 피보나치 수를 알려줄 수 있나요?")
```

> [!TIP]
> 에이전트를 컨텍스트 관리자(`with` 문)로 사용하면 에이전트가 작업을 완료한 직후 Docker 컨테이너가 정리되도록 보장합니다.
> 또는 에이전트의 `cleanup()` 메소드를 수동으로 호출할 수도 있습니다.

#### 고급 Docker 사용법[[advanced-docker-usage]]

Docker에서 다중 에이전트 시스템을 실행하려면 샌드박스에 사용자 정의 인터프리터를 설정해야 합니다.

Dockerfile을 설정하는 방법은 다음과 같습니다.

```dockerfile
FROM python:3.10-bullseye

# 빌드 의존성 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir smolagents && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# 제한된 권한으로 실행
USER nobody

# 기본 명령어
CMD ["python", "-c", "print('Container ready')"]
```

코드를 실행할 샌드박스 관리자를 생성합니다.

```python
import docker
import os
from typing import Optional

class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def create_container(self):
        try:
            image, build_logs = self.client.images.build(
                path=".",
                tag="agent-sandbox",
                rm=True,
                forcerm=True,
                buildargs={},
                # decode=True
            )
        except docker.errors.BuildError as e:
            print("Build error logs:")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            raise

        # 보안 제약 조건과 적절한 로깅으로 컨테이너 생성
        self.container = self.client.containers.run(
            "agent-sandbox",
            command="tail -f /dev/null",  # 컨테이너를 계속 실행 상태로 유지
            detach=True,
            tty=True,
            mem_limit="512m",
            cpu_quota=50000,
            pids_limit=100,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str) -> Optional[str]:
        if not self.container:
            self.create_container()

        # 컨테이너에서 코드 실행
        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody"
        )

        # 모든 출력 수집
        return exec_result.output.decode() if exec_result.output else None


    def cleanup(self):
        if self.container:
            try:
                self.container.stop()
            except docker.errors.NotFound:
                # 컨테이너가 이미 제거됨, 예상된 동작
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.container = None  # 참조 제거

# 사용 예시:
sandbox = DockerSandbox()

try:
    # 에이전트 코드 정의
    agent_code = """
import os
from smolagents import CodeAgent, InferenceClientModel

# 에이전트 초기화
agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[]
)

# 에이전트 실행
response = agent.run("20번째 피보나치 수는 무엇인가요?")
print(response)
"""

    # 샌드박스에서 코드 실행
    output = sandbox.run_code(agent_code)
    print(output)

finally:
    sandbox.cleanup()
```

### WebAssembly 설정[[webassembly-setup]]

WebAssembly(Wasm)는 코드를 안전한 샌드박스 환경에서 실행할 수 있게 해주는 바이너리 명령어 형식입니다.
빠르고 효율적이며 안전하게 설계되어 잠재적으로 신뢰할 수 없는 코드를 실행하는 데 탁월한 선택입니다.

`WasmExecutor`는 [Pyodide](https://pyodide.org/)와 [Deno](https://docs.deno.com/)를 사용합니다.

#### 설치[[installation]]

1. [시스템에 Deno 설치하기](https://docs.deno.com/runtime/getting_started/installation/)

#### WebAssembly에서 에이전트 실행하기: 빠른 시작[[running-your-agent-in-webassembly-quick-start]]

다음과 같이 에이전트 초기화에 `executor_type="wasm"`을 전달하기만 하면 됩니다.
```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(model=InferenceClientModel(), tools=[], executor_type="wasm")

agent.run("100번째 피보나치 수를 알려줄 수 있나요?")
```

### 샌드박스 모범 사례[[best-practices-for-sandboxes]]

이러한 핵심 사례는 E2B와 Docker 샌드박스 모두에 적용됩니다.

- 리소스 관리
  - 메모리 및 CPU 제한 설정
  - 실행 시간 초과 구현
  - 리소스 사용량 모니터링
- 보안
  - 최소한의 권한으로 실행
  - 불필요한 네트워크 접근 비활성화
  - 비밀 정보에 환경 변수 사용
- 환경
  - 의존성을 최소한으로 유지
  - 고정된 패키지 버전 사용
  - 기본 이미지를 사용하는 경우 정기적으로 업데이트

- 정리
  - 특히 Docker 컨테이너의 경우, 리소스를 잡아먹는 대롱거리는(dangling) 컨테이너가 생기지 않도록 항상 적절한 리소스 정리를 보장해야 합니다.

✨ 이러한 모범 사례를 따르고 적절한 정리 절차를 구현함으로써, 에이전트가 샌드박스 환경에서 안전하고 효율적으로 실행되도록 보장할 수 있습니다.

## 보안 접근 방식 비교[[comparing-security-approaches]]

앞서 다이어그램에서 설명했듯이, 두 샌드박싱 접근 방식은 서로 다른 보안적 의미를 가집니다.

### 접근 방식 1: 코드 스니펫만 샌드박스에서 실행[[approach-1-running-just-the-code-snippets-in-a-sandbox]]
- **장점**: 
  - 간단한 파라미터(`executor_type="e2b"` 또는 `executor_type="docker"`)로 설정하기 더 쉽습니다.
  - API 키를 샌드박스로 전송할 필요가 없습니다.
  - 로컬 환경을 더 잘 보호합니다.
- **단점**:
  - 다중 에이전트(관리형 에이전트)를 지원하지 않습니다.
  - 여전히 환경과 샌드박스 간에 상태를 전송해야 합니다.
  - 특정 코드 실행에 제한됩니다.

### 접근 방식 2: 전체 에이전트 시스템을 샌드박스에서 실행[[approach-2-running-the-entire-agentic-system-in-a-sandbox]]
- **장점**:
  - 다중 에이전트를 지원합니다.
  - 전체 에이전트 시스템을 완벽하게 격리합니다.
  - 복잡한 에이전트 아키텍처에 더 유연합니다.
- **단점**:
  - 더 많은 수동 설정이 필요합니다.
  - 민감한 API 키를 샌드박스로 전송해야 할 수 있습니다.
  - 더 복잡한 작업으로 인해 잠재적으로 더 높은 지연 시간이 발생할 수 있습니다.

보안 요구 사항과 애플리케이션의 요구 사항 사이에서 가장 균형이 잘 맞는 접근 방식을 선택하세요. 더 간단한 에이전트 아키텍처를 가진 대부분의 애플리케이션의 경우, 접근 방식 1이 보안과 사용 편의성 사이의 좋은 균형을 제공합니다. 완전한 격리가 필요한 더 복잡한 다중 에이전트 시스템의 경우, 설정이 더 복잡하지만 접근 방식 2가 더 나은 보안을 보장합니다.