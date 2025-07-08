import os

from smolagents import CodeAgent, LiteLLMRouterModel, WebSearchTool


# Make sure to setup the necessary environment variables!

llm_loadbalancer_model_list = [
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region_name": os.getenv("AWS_REGION"),
        },
    },
    # {
    #     "model_name": "model-group-2",
    #     "litellm_params": {
    #         "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #         "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    #         "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    #         "aws_region_name": os.getenv("AWS_REGION"),
    #     },
    # },
]


model = LiteLLMRouterModel(
    model_id="model-group-1",
    model_list=llm_loadbalancer_model_list,
    client_kwargs={"routing_strategy": "simple-shuffle"},
)
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True, return_full_result=True)

full_result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

print(full_result)
