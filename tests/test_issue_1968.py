from langchain_openai import ChatOpenAI
from smolagents import (
    CodeAgent, 
    LiteLLMModel, 
    OpenAIModel,
    WebSearchTool
)

model = ChatOpenAI(
            model_name="Qwen/Qwen3-30B-A3B-thinking-2507",
            api_key=os.getenv("MODELSCOPE_API_KEY"),
            base_url=os.getenv("MODELSCOPE_BASE_URL"),
            temperature=0.2,  # 随机性，0-1之间，值越高回复越多样
            max_tokens=1024,  # 最大生成token数
            streaming=True,
        )

print(model.invoke("你好"))
agent = CodeAgent(
    model=model,
    tools=[],
    planning_interval=3
)

print(agent.run("你好"))
