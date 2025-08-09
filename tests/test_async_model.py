import os
import sys
import typing
import unittest

from smolagents.asynchronous.models import AsyncAzureOpenAIServerModel


class TestModels(unittest.IsolatedAsyncioTestCase):
    async def test_sanity(self):
        self.assertTrue(True)

    async def test_client_has_expected_method(self):
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'create_client'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'generate'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, 'generate_stream'))
        self.assertTrue(hasattr(AsyncAzureOpenAIServerModel, '_apply_rate_limit'))

    async def test_generate_returns_string(self):
        client = AsyncAzureOpenAIServerModel(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_id="gpt4o",
        )

        generation_stream = await client(messages=[{"role": "user", "content": "Hello, world!"}])

        async for chunk in generation_stream:
            self.assertIsInstance(chunk, str)


if __name__ == '__main__':
    unittest.main()
