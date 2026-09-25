import sys
import unittest
from unittest.mock import MagicMock, patch
from smolagents.arxiv_tool import ArxivResearchTool

class TestArxivTool(unittest.TestCase):
    def test_search_papers(self):
        """
        Test paper search with mocked arXiv response.
        """
        # 1. Create Fake Result Object
        mock_paper = MagicMock()
        mock_paper.title = "Attention Is All You Need"
        mock_paper.published.strftime.return_value = "2017-06-12"
        mock_paper.pdf_url = "http://arxiv.org/pdf/1706.03762v5"
        mock_paper.summary = "The Transformer is a deep learning architecture..."

        # 2. Mock the Client
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_paper]

        # 3. Patch the module
        with patch.dict(sys.modules):
            # Create a mock module for 'arxiv'
            mock_arxiv_module = MagicMock()
            mock_arxiv_module.Client.return_value = mock_client
            mock_arxiv_module.SortCriterion.Relevance = "relevance"
            mock_arxiv_module.Search = MagicMock()
            
            sys.modules["arxiv"] = mock_arxiv_module
            
            # 4. Run Test
            tool = ArxivResearchTool()
            result = tool.forward("transformers")

            # 5. Verify
            self.assertIn("Attention Is All You Need", result)
            self.assertIn("http://arxiv.org/pdf/", result)
            self.assertIn("SUMMARY:", result)

    def test_missing_dependency(self):
        """Test graceful failure if arxiv is missing."""
        with patch.dict(sys.modules):
            if 'arxiv' in sys.modules:
                del sys.modules['arxiv']
            
            tool = ArxivResearchTool()
            result = tool.forward("query")
            self.assertIn("Please install 'arxiv'", result)

if __name__ == "__main__":
    unittest.main()
