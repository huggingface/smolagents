# 설치 옵션[[installation-options]]

`smolagents` 라이브러리는 pip를 사용하여 설치할 수 있습니다. 사용 가능한 다양한 설치 방법과 옵션을 소개합니다.

## 사전 요구사항[[prerequisites]]
- Python 3.10 이상
- Python 패키지 관리자: [`pip`](https://pip.pypa.io/en/stable/) 또는 [`uv`](https://docs.astral.sh/uv/)

## 가상 환경[[virtual-environment]]

`smolagents`를 Python 가상 환경 내에서 설치하는 것을 강력히 권장합니다.
가상 환경은 프로젝트의 의존성을 다른 Python 프로젝트와 시스템 Python 설치로부터 격리하여
버전 충돌을 방지하고 패키지 관리를 더욱 안정적으로 만들어줍니다.

<hfoptions id="virtual-environment">
<hfoption id="venv">

[`venv`](https://docs.python.org/3/library/venv.html) 사용:

```bash
python -m venv .venv
source .venv/bin/activate
```

</hfoption>
<hfoption id="uv">

[`uv`](https://docs.astral.sh/uv/) 사용:

```bash
uv venv .venv
source .venv/bin/activate
```

</hfoption>
</hfoptions>

## 기본 설치[[basic-installation]]

`smolagents` 핵심 라이브러리를 설치합니다:

<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install smolagents
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install smolagents
```
</hfoption>
</hfoptions>

## 추가 기능과 함께 설치[[installation-with-extras]]

`smolagents`는 필요에 따라 설치할 수 있는 여러 선택적 의존성(extras)을 제공합니다.
다음 구문을 사용하여 이러한 추가 기능을 설치할 수 있습니다:
<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[extra1,extra2]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[extra1,extra2]"
```
</hfoption>
</hfoptions>

### 도구[[tools]]
다음 추가 기능은 다양한 도구와 통합을 포함합니다:
<hfoptions id="installation">
<hfoption id="pip">
- **toolkit**: 일반적인 작업을 위한 기본 도구 세트를 설치합니다.
  ```bash
  pip install "smolagents[toolkit]"
  ```
- **mcp**: 외부 도구 및 서비스와 통합하기 위한 Model Context Protocol (MCP) 지원을 추가합니다.
  ```bash
  pip install "smolagents[mcp]"
  ```
</hfoption>
<hfoption id="uv">
- **toolkit**: 일반적인 작업을 위한 기본 도구 세트를 설치합니다.
  ```bash
  uv pip install "smolagents[toolkit]"
  ```
- **mcp**: 외부 도구 및 서비스와 통합하기 위한 Model Context Protocol (MCP) 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[mcp]"
  ```
</hfoption>
</hfoptions>

### 모델 통합[[model-integration]]
다음 추가 기능은 다양한 AI 모델 및 프레임워크와의 통합을 가능하게 합니다:
<hfoptions id="installation">
<hfoption id="pip">
- **openai**: OpenAI API 모델 지원을 추가합니다.
  ```bash
  pip install "smolagents[openai]"
  ```
- **transformers**: Hugging Face 트랜스포머 모델을 활성화합니다.
  ```bash
  pip install "smolagents[transformers]"
  ```
- **vllm**: 효율적인 모델 추론을 위한 VLLM 지원을 추가합니다.
  ```bash
  pip install "smolagents[vllm]"
  ```
- **mlx-lm**: MLX-LM 모델 지원을 활성화합니다.
  ```bash
  pip install "smolagents[mlx-lm]"
  ```
- **litellm**: 경량 모델 추론을 위한 LiteLLM 지원을 추가합니다.
  ```bash
  pip install "smolagents[litellm]"
  ```
- **bedrock**: AWS Bedrock 모델 지원을 활성화합니다.
  ```bash
  pip install "smolagents[bedrock]"
  ```
</hfoption>
<hfoption id="uv">
- **openai**: OpenAI API 모델 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[openai]"
  ```
- **transformers**: Hugging Face 트랜스포머 모델을 활성화합니다.
  ```bash
  uv pip install "smolagents[transformers]"
  ```
- **vllm**: 효율적인 모델 추론을 위한 VLLM 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[vllm]"
  ```
- **mlx-lm**: MLX-LM 모델 지원을 활성화합니다.
  ```bash
  uv pip install "smolagents[mlx-lm]"
  ```
- **litellm**: 경량 모델 추론을 위한 LiteLLM 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[litellm]"
  ```
- **bedrock**: AWS Bedrock 모델 지원을 활성화합니다.
  ```bash
  uv pip install "smolagents[bedrock]"
  ```
</hfoption>
</hfoptions>

### 멀티모달 기능[[multimodal-capabilities]]
다양한 미디어 유형 및 입력 처리를 위한 추가 기능:
<hfoptions id="installation">
<hfoption id="pip">
- **vision**: 이미지 처리 및 컴퓨터 비전 작업 지원을 추가합니다.
  ```bash
  pip install "smolagents[vision]"
  ```
- **audio**: 오디오 처리 기능을 활성화합니다.
  ```bash
  pip install "smolagents[audio]"
  ```
</hfoption>
<hfoption id="uv">
- **vision**: 이미지 처리 및 컴퓨터 비전 작업 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[vision]"
  ```
- **audio**: 오디오 처리 기능을 활성화합니다.
  ```bash
  uv pip install "smolagents[audio]"
  ```
</hfoption>
</hfoptions>

### 원격 실행[[remote-execution]]
코드를 원격으로 실행하기 위한 추가 기능:
<hfoptions id="installation">
<hfoption id="pip">
- **docker**: Docker 컨테이너에서 코드를 실행하는 지원을 추가합니다.
  ```bash
  pip install "smolagents[docker]"
  ```
- **e2b**: 원격 실행을 위한 E2B 지원을 활성화합니다.
  ```bash
  pip install "smolagents[e2b]"
  ```
</hfoption>
<hfoption id="uv">
- **docker**: Docker 컨테이너에서 코드를 실행하는 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[docker]"
  ```
- **e2b**: 원격 실행을 위한 E2B 지원을 활성화합니다.
  ```bash
  uv pip install "smolagents[e2b]"
  ```
</hfoption>
</hfoptions>

### 텔레메트리 및 사용자 인터페이스[[telemetry-and-user-interface]]
텔레메트리, 모니터링 및 사용자 인터페이스 구성 요소를 위한 추가 기능:
<hfoptions id="installation">
<hfoption id="pip">
- **telemetry**: 모니터링 및 추적 지원을 추가합니다.
  ```bash
  pip install "smolagents[telemetry]"
  ```
- **gradio**: 대화형 Gradio UI 구성 요소 지원을 추가합니다.
  ```bash
  pip install "smolagents[gradio]"
  ```
</hfoption>
<hfoption id="uv">
- **telemetry**: 모니터링 및 추적 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[telemetry]"
  ```
- **gradio**: 대화형 Gradio UI 구성 요소 지원을 추가합니다.
  ```bash
  uv pip install "smolagents[gradio]"
  ```
</hfoption>
</hfoptions>

### 전체 설치[[complete-installation]]
사용 가능한 모든 추가 기능을 설치하려면 다음을 사용할 수 있습니다:
<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[all]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[all]"
```
</hfoption>
</hfoptions>

## 설치 확인[[verifying-installation]]
설치 후, 다음 코드를 실행해 `smolagents`가 올바르게 설치되었는지 확인할 수 있습니다:
```python
import smolagents
print(smolagents.__version__)
```

## 다음 단계[[next-steps]]
`smolagents`를 성공적으로 설치했다면 다음을 수행할 수 있습니다:
- [안내서](./guided_tour)를 따라 기본 개념을 배워보세요.
- 실용적인 예제를 보고 싶다면 [사용법 가이드](./examples/text_to_sql)를 살펴보세요.
- 고수준 설명을 보려면 [개념 가이드](./conceptual_guides/intro_agents)를 읽어보세요.
- 에이전트 구축에 대한 심화 튜토리얼은 [튜토리얼](./tutorials/building_good_agents)를 확인해보세요.
- 클래스와 함수에 대한 자세한 정보를 확인하고 싶으시면 [API 레퍼런스](./reference/index)를 살펴보세요.
