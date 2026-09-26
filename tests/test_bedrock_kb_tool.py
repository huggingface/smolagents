# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from unittest.mock import MagicMock, patch

from smolagents.bedrock_kb_tool import BedrockKnowledgeBaseTool


class TestBedrockKnowledgeBaseTool(unittest.TestCase):
    """Tests for the BedrockKnowledgeBaseTool."""

    def test_tool_attributes(self):
        """Test tool has required attributes."""
        tool = BedrockKnowledgeBaseTool(knowledge_base_id="TEST123456")
        assert tool.name == "bedrock_knowledge_base"
        assert tool.output_type == "string"
        assert "query" in tool.inputs
        assert tool.inputs["query"]["type"] == "string"

    def test_init_with_params(self):
        """Test initialization with explicit parameters."""
        tool = BedrockKnowledgeBaseTool(
            knowledge_base_id="ABCDEFGHIJ",
            region_name="us-west-2",
            number_of_results=10,
            use_agentic_retrieval=True,
        )
        assert tool.knowledge_base_id == "ABCDEFGHIJ"
        assert tool.region_name == "us-west-2"
        assert tool.number_of_results == 10
        assert tool.use_agentic_retrieval is True

    @patch.dict("os.environ", {"KNOWLEDGE_BASE_ID": "ENV_KB_ID", "AWS_REGION": "eu-west-1"})
    def test_init_from_env_vars(self):
        """Test initialization from environment variables."""
        tool = BedrockKnowledgeBaseTool()
        assert tool.knowledge_base_id == "ENV_KB_ID"
        assert tool.region_name == "eu-west-1"

    def test_no_kb_id_returns_error(self):
        """Test that missing KB ID returns an error message."""
        tool = BedrockKnowledgeBaseTool(knowledge_base_id="")
        tool.is_initialized = True
        tool._client = MagicMock()
        result = tool.forward("test query")
        assert "Error" in result or "No knowledge_base_id" in result

    def test_managed_retrieve(self):
        """Test retrieval with managed search configuration."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Managed KB provides fully managed RAG."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
                    "score": 0.95,
                },
                {
                    "content": {"text": "No vector store needed."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                    "score": 0.87,
                },
            ]
        }

        tool = BedrockKnowledgeBaseTool(
            knowledge_base_id="TEST123456",
            region_name="us-west-2",
            use_agentic_retrieval=False,
        )
        tool._client = mock_client
        tool.is_initialized = True
        result = tool.forward("What is managed KB?")

        # Verify retrieve was called with managedSearchConfiguration
        mock_client.retrieve.assert_called_once()
        call_args = mock_client.retrieve.call_args
        assert call_args.kwargs["knowledgeBaseId"] == "TEST123456"
        assert "managedSearchConfiguration" in call_args.kwargs["retrievalConfiguration"]
        assert call_args.kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 5

        # Verify result contains retrieved content
        assert "Managed KB provides fully managed RAG" in result
        assert "s3://bucket/doc.pdf" in result

    def test_agentic_retrieve_fallback(self):
        """Test that agentic retrieval falls back to managed retrieve on error."""
        mock_client = MagicMock()
        # Agentic fails
        mock_client.agentic_retrieve_stream.side_effect = AttributeError("Not available")
        # Managed succeeds
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Fallback result."},
                    "location": {"s3Location": {"uri": "s3://bucket/fallback.pdf"}},
                    "score": 0.9,
                },
            ]
        }

        tool = BedrockKnowledgeBaseTool(
            knowledge_base_id="TEST123456",
            use_agentic_retrieval=True,
        )
        tool._client = mock_client
        tool.is_initialized = True
        result = tool.forward("Complex query needing decomposition")

        # Should have fallen back to managed retrieve
        assert "Fallback result" in result

    def test_empty_results(self):
        """Test handling of empty results."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {"retrievalResults": []}

        tool = BedrockKnowledgeBaseTool(
            knowledge_base_id="TEST123456",
            use_agentic_retrieval=False,
        )
        tool._client = mock_client
        tool.is_initialized = True
        result = tool.forward("query with no matches")

        assert "No relevant documents found" in result

    def test_retrieve_error_handling(self):
        """Test error handling when retrieve API fails."""
        mock_client = MagicMock()
        mock_client.retrieve.side_effect = Exception("Service unavailable")

        tool = BedrockKnowledgeBaseTool(
            knowledge_base_id="TEST123456",
            use_agentic_retrieval=False,
        )
        tool._client = mock_client
        tool.is_initialized = True
        result = tool.forward("test query")

        assert "Error" in result


if __name__ == "__main__":
    unittest.main()
