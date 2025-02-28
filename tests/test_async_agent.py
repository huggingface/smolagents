from smolagents import AsyncCodeAgent
import argparse
import asyncio

async def main(args: argparse.Namespace):
    agent = AsyncCodeAgent(tools=[], add_base_tools=True)

    question = "롯데이노베이트 2024년 영업이익 알려줘"

    response = await agent.run(question)

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    asyncio.run(main(args))