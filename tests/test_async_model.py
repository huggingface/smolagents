import os
import sys
import typing
import unittest

from smolagents.asynchronous import AsyncOpenaAIClient


class TestModels(unittest.IsolatedAsyncioTestCase):
    async def test_sanity(self):
        self.assertTrue(True)

    async def test_client_has_expected_method(self):
        self.assertTrue(hasattr(AzureOpenAIClient, '_prepare_inputs'))
        self.assertTrue(hasattr(AzureOpenAIClient, 'generate'))
        self.assertTrue(hasattr(AzureOpenAIClient, '__call__'))

    async def test_generate_returns_string(self):
        client = AzureOpenAIClient(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            model_name="gpt4o",
        )

        generation_stream = client(inputs="Hello, who are you?")
        self.assertIsInstance(generation_stream, typing.AsyncGenerator)

        async for chunk in generation_stream:
            self.assertIsInstance(chunk, str)


if __name__ == '__main__':
    unittest.main()
