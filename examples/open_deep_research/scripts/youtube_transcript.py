def fetch_transcript_srt(video_id: str) -> str:
    """Transcript for one video, as SRT.

    youtube-transcript-api 1.0 removed the static get_transcript helper. 0.6 still
    has it, and each release's SRTFormatter expects that release's transcript shape.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import SRTFormatter

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    else:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    return SRTFormatter().format_transcript(transcript)
