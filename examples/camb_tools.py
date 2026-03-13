"""Example usage of Camb.ai tools with smolagents.

Requires:
    - CAMB_API_KEY in .env file or environment
    - Sabrina Carpenter audio clip at ../yt-dlp/voices/original/sabrina-original-clip.mp3

Usage:
    .venv/bin/python examples/camb_tools.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from smolagents import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTextToSpeechTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
)

API_KEY = os.environ.get("CAMB_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CAMB_API_KEY in .env or environment")

_DEFAULT_SAMPLE = Path(__file__).resolve().parents[2] / "yt-dlp" / "voices" / "original" / "sabrina-original-clip.mp3"
AUDIO_SAMPLE = os.environ.get("CAMB_AUDIO_SAMPLE", str(_DEFAULT_SAMPLE))
if not Path(AUDIO_SAMPLE).exists():
    raise RuntimeError(f"Audio sample not found at {AUDIO_SAMPLE}")


def play(path: str) -> None:
    """Play audio with afplay (macOS)."""
    if sys.platform == "darwin":
        print(f"  Playing: {path}")
        subprocess.run(["afplay", path], check=False)
    else:
        print(f"  Audio at: {path}")


def example_tts():
    """1. Text-to-Speech: convert text to audio."""
    tool = CambTextToSpeechTool(api_key=API_KEY)
    path = tool("Hello from Camb AI and smolagents! This is a text to speech test.")
    print(f"  Audio saved to: {path}")
    play(path)


def example_list_voices():
    """2. List Voices: show available voices."""
    tool = CambVoiceListTool(api_key=API_KEY)
    result = tool(language_filter=None)
    voices = json.loads(result)
    print(f"  Found {len(voices)} voices")
    for v in voices[:3]:
        print(f"    - {v['name']} (id={v['id']}, gender={v['gender']})")


def example_translation():
    """3. Translation: translate text between languages."""
    tool = CambTranslationTool(api_key=API_KEY)
    result = tool("Hello, how are you?", source_language=1, target_language=2)
    print(f"  Translated: {result}")


def example_transcription():
    """4. Transcription: transcribe the Sabrina clip."""
    tool = CambTranscriptionTool(api_key=API_KEY)
    result = tool(audio_file_path=AUDIO_SAMPLE, language=1)
    parsed = json.loads(result)
    print(f"  Text: {parsed.get('text', result[:200])}")
    segments = parsed.get("segments", [])
    print(f"  Segments: {len(segments)}")


def example_translated_tts():
    """5. Translated TTS: translate and speak in one step."""
    tool = CambTranslatedTTSTool(api_key=API_KEY)
    path = tool("Hello, how are you?", source_language=1, target_language=2)
    print(f"  Audio saved to: {path}")
    play(path)


def example_text_to_sound():
    """6. Text-to-Sound: generate sound effects from a prompt."""
    tool = CambTextToSoundTool(api_key=API_KEY)
    path = tool("gentle rain on a rooftop")
    print(f"  Audio saved to: {path}")
    play(path)


def example_voice_clone():
    """7. Voice Clone: clone Sabrina's voice and speak with it."""
    tool = CambVoiceCloneTool(api_key=API_KEY)
    result = tool(voice_name="smolagents_sabrina", audio_file_path=AUDIO_SAMPLE, gender=2)
    data = json.loads(result)
    voice_id = data["voice_id"]
    print(f"  Cloned voice ID: {voice_id}")

    # Now speak with the cloned voice
    tts = CambTextToSpeechTool(api_key=API_KEY, voice_id=voice_id)
    path = tts("Hello! This is Sabrina, cloned with Camb AI and smolagents.")
    print(f"  Speaking with cloned voice...")
    play(path)


def example_audio_separation():
    """8. Audio Separation: separate vocals from background."""
    tool = CambAudioSeparationTool(api_key=API_KEY)
    result = tool(audio_file_path=AUDIO_SAMPLE)
    parsed = json.loads(result)
    print(f"  Status: {parsed['status']}")
    print(f"  Foreground: {parsed['vocals']}")
    print(f"  Background: {parsed['background']}")


if __name__ == "__main__":
    examples = [
        example_tts,
        example_list_voices,
        example_translation,
        example_transcription,
        example_translated_tts,
        example_text_to_sound,
        example_voice_clone,
        example_audio_separation,
    ]
    for fn in examples:
        print(f"\n--- {fn.__doc__} ---")
        try:
            fn()
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
