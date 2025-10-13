import os
import unittest

import smolagents
from smolagents.asynchronous.models import AsyncAzureOpenAIServerModel
from smolagents.asynchronous.agents import AsyncCodeAgent

class TestModels(unittest.IsolatedAsyncioTestCase):
    async def test_sanity(self):
        self.assertTrue(True)

    async def test_client_has_expected_method(self):
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'create_client'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'generate'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'generate_stream'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, '_apply_rate_limit'))

    async def test_code_agent_generate(self):
        client = AsyncAzureOpenAIServerModel(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_id="gpt4d1",
        )

        agent = AsyncCodeAgent(model=client, tools=[])
        result = await agent.run("Hello, world!", stream=False)
        print(result)

    async def test_code_agent_generate_stream(self):
        client = AsyncAzureOpenAIServerModel(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_id="gpt4d1",
        )

        agent = AsyncCodeAgent(model=client, tools=[])
        result = await agent.run("Hello, world!", stream=True)
        async for line in result:
            print("Chunk:",line)

    async def test_code_agent_generate_with_tools(self):
        from smolagents.default_tools import DuckDuckGoSearchTool
        client = AsyncAzureOpenAIServerModel(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_id="gpt4d1",
        )

        tools = [DuckDuckGoSearchTool()]
        agent = AsyncCodeAgent(model=client, tools=tools)
        result = await agent.run("롯데 뉴스 알려줘", stream=False)
        print(result)

    async def test_code_agent_generate_with_tools_stream(self):
        from smolagents.default_tools import DuckDuckGoSearchTool
        client = AsyncAzureOpenAIServerModel(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_id="gpt4d1",
        )

        tools = [DuckDuckGoSearchTool()]
        agent = AsyncCodeAgent(model=client, tools=tools)
        result = await agent.run("News for the LLM", stream=True)
        async for line in result:
            print("Chunk:",line)




if __name__ == '__main__':
    unittest.main()
