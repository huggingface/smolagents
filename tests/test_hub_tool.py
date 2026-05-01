import unittest
from unittest.mock import MagicMock, patch
from smolagents.hub_tool import HubSearchTool

class TestHubTool(unittest.TestCase):
    def test_hub_search(self):
        """
        Test Hub search with mocked HfApi response.
        """
        # 1. Mock Model Object
        mock_model = MagicMock()
        mock_model.modelId = "meta-llama/Llama-3-8b"
        mock_model.likes = 10000
        mock_model.downloads = 500000

        # 2. Mock Dataset Object
        mock_dataset = MagicMock()
        mock_dataset.id = "common_voice"
        mock_dataset.likes = 500
        mock_dataset.downloads = 20000

        # 3. Mock the API
        mock_api = MagicMock()
        mock_api.list_models.return_value = [mock_model]
        mock_api.list_datasets.return_value = [mock_dataset]

        # 4. Patch and Run
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            tool = HubSearchTool()
            result = tool.forward("llama")
            
            # 5. Verify
            self.assertIn("meta-llama/Llama-3-8b", result)
            self.assertIn("common_voice", result)
            self.assertIn("https://huggingface.co/", result)

if __name__ == "__main__":
    unittest.main()
