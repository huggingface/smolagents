import re
from .tools import Tool


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript_tool"
    description = "Extracts the transcript (subtitles) from a YouTube video URL using youtube-transcript-api. Input: url (str). Output: transcript (str)."
    inputs = {
        "url": {
            "type": "string",
            "description": "The full YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the YouTube Transcript tool.
        """
        super().__init__(**kwargs)

    def forward(self, url: str) -> str:
        """
        Extracts the transcript (subtitles) from a YouTube video.
        
        Args:
            url: The full YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ).
        """
        # Lazy Import: Keeps the library lightweight
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return "Error: Please install 'youtube-transcript-api' to use this tool."

        try:
            # 1. robust Regex to find Video ID (handles regular, short, and embed URLs)
            video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
            if not video_id_match:
                return "Error: Could not extract a valid Video ID from the URL."
            
            video_id = video_id_match.group(1)

            # 2. Fetch Transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # 3. Format into a single string
            full_text = " ".join([item['text'] for item in transcript_list])
            
            return f"Transcript for Video ID {video_id}:\n\n{full_text}"

        except Exception as e:
            if "Subtitles are disabled" in str(e):
                return "Error: This video does not have subtitles/transcripts enabled."
            return f"Error fetching transcript: {str(e)}"
