# Agentic RAG[[agentic-rag]]

[[open-in-colab]]

## RAG(검색 증강 생성) 소개[[introduction-to-retrieval-augmented-generation-rag]]

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 대규모 언어 모델의 능력과 외부 지식 검색을 결합하여 더 정확하고 사실에 기반을 두며 문맥에 맞는 응답을 생성합니다. RAG의 핵심은 "대규모 언어 모델을 사용해 사용자 쿼리에 답변을 제공하되, 지식 베이스에서 검색된 정보에 기반하여 답변하는 것"입니다.

### RAG를 사용하는 이유[[why-use-rag]]

RAG는 기본 대규모 언어 모델이나 미세 조정된 모델을 사용하는 것에 비해 다음과 같은 몇 가지 중요한 장점을 제공합니다.

1. **사실 기반 생성**: 답변의 근거를 검색 결과에 두어 환각 현상을 줄입니다.
2. **도메인 특화**: 모델을 다시 훈련시키지 않고도 특정 도메인의 지식을 제공합니다.
3. **최신 지식 반영**: 모델의 훈련 시점 이후의 정보에도 접근할 수 있습니다.
4. **투명성**: 생성된 콘텐츠의 출처를 인용할 수 있습니다.
5. **제어**: 모델이 접근할 수 있는 정보를 세밀하게 제어할 수 있습니다.

### 전통적인 RAG의 한계[[limitations-of-traditional-rag]]

이러한 장점에도 불구하고, 전통적인 RAG 접근 방식은 다음과 같은 몇 가지 문제가 있습니다.

- **단일 검색 단계**: 초기 검색 결과가 좋지 않으면 최종 생성 결과의 품질이 저하됩니다.
- **쿼리-문서 불일치**: 사용자 쿼리(주로 질문)가 답변을 포함하는 문서(주로 서술문)와 잘 일치하지 않을 수 있습니다.
- **제한된 추론**: 단순한 RAG 파이프라인은 다단계 논리적 추론이나 쿼리 정제를 허용하지 않습니다.
- **컨텍스트 윈도우 제약**: 검색된 문서는 모델의 컨텍스트 윈도우 크기에 맞춰야 합니다.

## Agentic RAG: 더 강력한 접근 방식[[agentic-rag-a-more-powerful-approach]]

**Agentic RAG** 시스템, 즉 검색 능력을 갖춘 에이전트를 구현함으로써 이러한 한계를 극복할 수 있습니다. 이 접근 방식은 RAG를 경직된 파이프라인에서 논리적 추론 중심의 상호작용적 프로세스로 탈바꿈시키는 방식입니다.

### Agentic RAG의 주요 장점[[key-benefits-of-agentic-rag]]

검색 도구를 갖춘 에이전트는 다음을 수행할 수 있습니다.

1. ✅ **최적화된 쿼리 생성**: 에이전트는 사용자 질문을 검색에 적합한 쿼리로 변환할 수 있습니다.
2. ✅ **다중 검색 수행**: 에이전트는 필요에 따라 반복적으로 정보를 검색할 수 있습니다.
3. ✅ **검색 결과 기반 논리적 추론**: 에이전트는 여러 소스의 정보를 분석, 종합하고 결론을 도출할 수 있습니다.
4. ✅ **자체 평가 및 개선**: 에이전트는 검색 결과를 평가하고 접근 방식을 조정할 수 있습니다.

이 접근 방식은 다음과 같은 Agentic RAG 기술을 자연스럽게 구현합니다.
- **가상 문서 임베딩(HyDE)**: 사용자 쿼리를 직접 사용하는 대신, 에이전트가 검색에 최적화된 쿼리를 생성합니다 ([논문 참조](https://huggingface.co/papers/2212.10496))
- **자가 쿼리 정제**: 에이전트가 초기 결과를 분석하고 정제된 쿼리로 후속 검색을 수행할 수 있습니다 ([기술 참조](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/))

## Agentic RAG 시스템 구축하기[[building-an-agentic-rag-system]]

이제 단계별로 Agentic RAG 시스템을 구축해 보겠습니다. 이 예제에서는 허깅 페이스 Transformers 라이브러리 설명서를 검색해 질문에 답할 수 있는 에이전트를 만들어 보겠습니다.

아래 코드 스니펫을 따라 하거나, smolagents GitHub 리포지토리에서 전체 예제를 확인할 수 있습니다: [examples/rag.py](https://github.com/huggingface/smolagents/blob/main/examples/rag.py).

### 1단계: 필요한 의존성 설치하기[[step-1-install-required-dependencies]]

먼저, 필요한 패키지를 설치해야 합니다.

```bash
pip install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 --upgrade
```

허깅 페이스의 추론 API를 사용하려면 API 토큰을 설정해야 합니다.

```python
# 환경 변수 로드 (HF_TOKEN 포함)
from dotenv import load_dotenv
load_dotenv()
```

### 2단계: 지식 베이스 준비하기[[step-2-prepare-the-knowledge-base]]

허깅 페이스 설명서가 포함된 데이터 세트를 불러와 검색에 사용할 준비를 해보겠습니다.

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# 허깅 페이스 설명서 데이터 세트 로드
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Transformers 라이브러리 설명서만 포함하도록 필터링
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# 데이터 세트 항목을 메타데이터가 있는 Document 객체로 변환
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# 더 나은 검색을 위해 문서를 작은 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 청크당 문자 수
    chunk_overlap=50,  # 컨텍스트 유지를 위한 청크 간 중첩
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # 분할 우선순위
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")
```

### 3단계: 검색 도구 만들기[[step-3-create-a-retriever-tool]]

이제 에이전트가 지식 베이스에서 정보를 검색하는 데 사용할 수 있는 사용자 정의 도구를 만들어 보겠습니다.

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "의미 기반 검색을 사용하여 쿼리에 답변하는 데 가장 관련성이 높은 transformers 설명서 부분을 검색합니다."
    inputs = {
        "query": {
            "type": "string",
            "description": "수행할 쿼리입니다. 대상 문서와 의미적으로 가까워야 합니다. 질문보다는 긍정문을 사용하세요.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # 처리된 문서로 검색기 초기화
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # 가장 관련성 높은 상위 10개 문서 반환
        )

    def forward(self, query: str) -> str:
        """제공된 쿼리를 기반으로 검색을 실행합니다."""
        assert isinstance(query, str), "Your search query must be a string"

        # 관련 문서 검색
        docs = self.retriever.invoke(query)

        # 가독성을 위해 검색된 문서 형식 지정
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# 처리된 문서로 검색 도구 초기화
retriever_tool = RetrieverTool(docs_processed)
```

> [!TIP]
> 단순성과 속도를 위해 어휘 검색 방식인 BM25를 사용하고 있습니다. 실제 서비스 환경에서는 검색 품질을 높이기 위해 임베딩을 활용한 의미 기반 검색을 사용하는 것이 좋습니다. 고품질 임베딩 모델은 [MTEB 리더보드](https://huggingface.co/spaces/mteb/leaderboard)에서 확인하세요.

### 4단계: 고급 검색 에이전트 만들기[[step-4-create-an-advanced-retrieval-agent]]

다음으로 앞서 만든 검색 도구를 활용해 질문에 답할 수 있는 에이전트를 구성해 봅시다.

```python
from smolagents import InferenceClientModel, CodeAgent

# 검색 도구로 에이전트 초기화
agent = CodeAgent(
    tools=[retriever_tool],  # 에이전트가 사용할 수 있는 도구 목록
    model=InferenceClientModel(),  # 기본 모델 "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_steps=4,  # 논리적 추론 단계 수 제한
    verbosity_level=2,  # 에이전트의 상세한 논리적 추론 과정 표시
)

# 특정 모델을 사용하려면 다음과 같이 지정할 수 있습니다:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
```

> [!TIP]
> Inference Provider는 서버리스 추론 파트너가 제공하는 수백 개의 모델에 대한 액세스를 제공합니다. 지원되는 제공업체 목록은 [여기](https://huggingface.co/docs/inference-providers/index)에서 찾을 수 있습니다.

### 5단계: 에이전트를 실행하여 질문에 답하기[[step-5-run-the-agent-to-answer-questions]]

마지막으로 에이전트를 실행해 Transformers 관련 질문에 답해 보겠습니다.

```python
# 정보 검색이 필요한 질문하기
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# 에이전트를 실행하여 답변 얻기
agent_output = agent.run(question)

# 최종 답변 표시
print("\nFinal answer:")
print(agent_output)
```

## Agentic RAG의 실제 적용 사례[[practical-applications-of-agentic-rag]]

Agentic RAG 시스템은 다양한 사용 사례에 적용될 수 있습니다.

1. **기술 문서 지원**: 사용자가 복잡한 기술 문서를 탐색하는 데 도움을 줍니다.
2. **연구 논문 분석**: 과학 논문에서 정보를 추출하고 종합합니다.
3. **법률 문서 검토**: 법률 문서에서 관련 판례와 조항을 찾습니다.
4. **고객 지원**: 제품 설명서와 지식 베이스를 기반으로 질문에 답변합니다.
5. **교육 튜터링**: 교과서와 학습 자료를 기반으로 설명을 제공합니다.

## 결론[[conclusion]]

Agentic RAG는 전통적인 RAG 파이프라인을 뛰어넘는 중요한 발전을 의미합니다. 대형 언어 모델 에이전트의 추론 능력과 검색 시스템의 사실 기반을 결합함으로써, 우리는 더 강력하고 유연하며 정확한 정보 시스템을 구축할 수 있습니다.

저희가 보여드린 접근 방식은 다음과 같은 특징이 있습니다:
- 단일 단계 검색의 한계를 극복합니다.
- 지식 베이스와의 상호작용이 더 자연스러워집니다.
- 자체 평가와 쿼리 정제를 통해 지속해서 개선할 수 있는 프레임워크를 제공합니다.

자신만의 Agentic RAG 시스템을 구축할 때에는, 다양한 검색 방법과 에이전트 아키텍처, 지식 소스를 실험하며 사용 사례에 최적화된 구성을 찾아보세요.