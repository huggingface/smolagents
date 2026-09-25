import sys
import unittest
from unittest.mock import MagicMock, patch
from smolagents.youtube_transcript_tool import YouTubeTranscriptTool

class TestYouTubeTranscriptTool(unittest.TestCase):
    def test_transcript_extraction(self):
        """
        Test that the tool correctly parses the transcript 
        WITHOUT actually connecting to YouTube (Mocking).
        """
        
        # 1. Create a Fake YouTube API
        mock_api = MagicMock()
        mock_api.get_transcript.return_value = [
            {'text': 'Hello world', 'start': 0.0},
            {'text': 'This is a test', 'start': 1.0}
        ]
        
        # 2. Inject the fake API into the system modules
        # This prevents 'ImportError' during testing
        with patch.dict(sys.modules, {'youtube_transcript_api': mock_api}):
            # We also need to patch the class inside the module
            mock_api.YouTubeTranscriptApi = mock_api
            
            tool = YouTubeTranscriptTool()
            # Fake URL (doesn't matter, response is mocked)
            result = tool.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            
            # 3. Verify Logic
            self.assertIn("Hello world This is a test", result)
            self.assertIn("dQw4w9WgXcQ", result)

    def test_missing_dependency(self):
        """Test that it fails gracefully if the library is missing."""
        # Forcefully remove the library from modules to simulate it missing
        with patch.dict(sys.modules):
            if 'youtube_transcript_api' in sys.modules:
                del sys.modules['youtube_transcript_api']
            
            tool = YouTubeTranscriptTool()
            result = tool.forward("https://youtube.com/watch?v=123")
            self.assertIn("Please install 'youtube-transcript-api'", result)

if __name__ == "__main__":
    unittest.main()
