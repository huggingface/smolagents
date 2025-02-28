from smolagents import AsyncCodeAgent, AsyncAzureOpenAIServerModel
import argparse
import asyncio


async def main(args: argparse.Namespace):
    model = AsyncAzureOpenAIServerModel(
        model_id=args.model_id,
        api_key=args.api_key,
        api_version=args.api_version,
        azure_endpoint=args.azure_endpoint
    )


    agent = AsyncCodeAgent(
        model=model,
        tools=[],
        add_base_tools=True
    )

    question = "롯데이노베이트 2024년 영업이익 알려줘"

    response = await agent.run(question)

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="gpt4o", help="The model deployment name to use when connecting (e.g. 'gpt-4o-mini').")
    parser.add_argument("--api_key", default="", help="The API key for the model.")
    parser.add_argument("--api_version", default="2024-12-01-preview", help="The API version.")
    parser.add_argument("--azure_endpoint", default="https://ai-tech-openai.openai.azure.com", help="The Azure endpoint.")
    args = parser.parse_args()
    asyncio.run(main(args))