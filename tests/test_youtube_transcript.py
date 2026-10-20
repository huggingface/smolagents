import importlib.util
from pathlib import Path

import youtube_transcript_api
from youtube_transcript_api.formatters import SRTFormatter


_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "open_deep_research" / "scripts" / "youtube_transcript.py"
)
_spec = importlib.util.spec_from_file_location("open_deep_research_youtube_transcript", _HELPER_PATH)
assert _spec is not None and _spec.loader is not None
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)


def test_fetch_is_used_when_get_transcript_is_gone(monkeypatch):
    class Api:
        def fetch(self, video_id):
            assert video_id == "dQw4w9WgXcQ"
            return "fetched"

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", Api)
    monkeypatch.setattr(SRTFormatter, "format_transcript", lambda self, transcript: f"srt:{transcript}")

    assert _helper.fetch_transcript_srt("dQw4w9WgXcQ") == "srt:fetched"


def test_older_api_still_uses_get_transcript(monkeypatch):
    class Api:
        @staticmethod
        def get_transcript(video_id):
            assert video_id == "abc"
            return [{"text": "hi", "start": 0.0, "duration": 1.0}]

        def fetch(self, video_id):
            raise AssertionError("0.6 should keep using get_transcript")

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", Api)
    monkeypatch.setattr(SRTFormatter, "format_transcript", lambda self, transcript: "old-srt")

    assert _helper.fetch_transcript_srt("abc") == "old-srt"
