#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import os
import tempfile
import time

from .tools import Tool


class _CambToolBase:
    """Shared mixin providing Camb.ai API helpers for all camb tools."""

    def _camb_init(
        self,
        api_key: str | None = None,
        base_url: str = "https://client.camb.ai/apis",
        timeout: float = 60.0,
        max_poll_attempts: int = 60,
        poll_interval: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Camb.ai API key is required. Set it via 'api_key' parameter or CAMB_API_KEY environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval

    def _get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _poll_task(self, task_endpoint: str, task_id: str | int) -> dict:
        """Poll until an async task completes. Returns the status dict."""
        import requests

        for _ in range(self.max_poll_attempts):
            resp = requests.get(
                f"{self.base_url}/{task_endpoint}/{task_id}",
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            status = resp.json()
            status_value = status.get("status", "")
            if status_value in ("completed", "SUCCESS"):
                return status
            if status_value in ("failed", "FAILED", "error"):
                raise RuntimeError(f"Camb.ai task failed: {status}")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Camb.ai task {task_id} did not complete within {self.max_poll_attempts * self.poll_interval} seconds"
        )

    @staticmethod
    def _detect_audio_suffix(audio_data: bytes) -> str:
        """Detect audio format from magic bytes."""
        if audio_data[:4] == b"RIFF":
            return ".wav"
        if audio_data[:3] == b"ID3" or audio_data[:2] == b"\xff\xfb":
            return ".mp3"
        if audio_data[:4] == b"fLaC":
            return ".flac"
        if audio_data[:4] == b"OggS":
            return ".ogg"
        return ".wav"

    @staticmethod
    def _save_audio_to_temp(audio_data: bytes, suffix: str = ".wav") -> str:
        """Save audio bytes to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            return f.name


class CambTextToSpeechTool(_CambToolBase, Tool):
    """Text-to-speech tool that converts text to audio using the Camb.ai API.

    Generates speech from text input and saves it to a temporary WAV file.
    Supports configurable voice, language, and speech model.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.
        voice_id (`int`, default `147320`): Numeric ID of the voice to use. Use `CambVoiceListTool` to discover available voices.
        language (`str`, default `"en-us"`): Language code for speech synthesis (e.g. `"en-us"`, `"fr-fr"`).
        speech_model (`str`, default `"mars-flash"`): Speech model identifier.

    Examples:
        ```python
        >>> from smolagents import CambTextToSpeechTool
        >>> tts = CambTextToSpeechTool()
        >>> audio_path = tts("Hello, how are you?")
        >>> print(audio_path)  # /tmp/...wav
        ```
    """

    name = "camb_text_to_speech"
    description = "Converts text to speech using Camb.ai. Returns the file path to the generated audio WAV file."
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to convert to speech (3-3000 characters).",
        }
    }
    output_type = "string"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: int = 147320,
        language: str = "en-us",
        speech_model: str = "mars-flash",
        **kwargs,
    ):
        self.voice_id = voice_id
        self.language = language
        self.speech_model = speech_model
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, text: str) -> str:
        import requests

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speech_model": self.speech_model,
            "output_configuration": {"format": "wav"},
        }
        response = requests.post(
            f"{self.base_url}/tts-stream",
            json=payload,
            headers=self._get_headers(),
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        audio_data = b"".join(response.iter_content(chunk_size=8192))
        suffix = self._detect_audio_suffix(audio_data)
        return self._save_audio_to_temp(audio_data, suffix=suffix)


class CambTranslationTool(_CambToolBase, Tool):
    """Text translation tool that translates between 140+ languages using the Camb.ai API.

    Translates text from a source language to a target language. Languages are specified
    as integer codes (e.g. 1 for English, 2 for Spanish).

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambTranslationTool
        >>> translator = CambTranslationTool()
        >>> result = translator("Hello, how are you?", source_language=1, target_language=2)
        >>> print(result)
        ```
    """

    name = "camb_translation"
    description = (
        "Translates text between 140+ languages using Camb.ai. "
        "Source and target languages are specified as integer language codes "
        "(e.g. 1 for English)."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to translate.",
        },
        "source_language": {
            "type": "integer",
            "description": "Source language code (integer, e.g. 1 for English).",
        },
        "target_language": {
            "type": "integer",
            "description": "Target language code (integer, e.g. 2 for Spanish).",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, text: str, source_language: int, target_language: int) -> str:
        import requests

        payload = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        response = requests.post(
            f"{self.base_url}/translation/stream",
            json=payload,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        # API returns plain text (not JSON)
        return response.text


class CambTranscriptionTool(_CambToolBase, Tool):
    """Audio transcription tool that converts speech to text using the Camb.ai API.

    Transcribes audio files to text with speaker detection and timestamped segments.
    Returns the result as a JSON string.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambTranscriptionTool
        >>> transcriber = CambTranscriptionTool()
        >>> result = transcriber("recording.wav", language=1)
        >>> print(result)  # JSON with text, segments, and speakers
        ```
    """

    name = "camb_transcription"
    description = (
        "Transcribes audio files to text with speaker detection using Camb.ai. "
        "Returns JSON with transcribed text, segments, and speakers."
    )
    inputs = {
        "audio_file_path": {
            "type": "string",
            "description": "Path to the audio file to transcribe.",
        },
        "language": {
            "type": "integer",
            "description": "Language code (integer, e.g. 1 for English).",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, audio_file_path: str, language: int) -> str:
        import requests

        with open(audio_file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/transcribe",
                headers={"x-api-key": self.api_key},
                files={"media_file": f},
                data={"language": str(language)},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id") or task_data.get("id")
        if not task_id:
            raise RuntimeError(f"Camb.ai API did not return a task ID: {task_data}")

        status = self._poll_task("transcribe", task_id)
        run_id = status.get("run_id")
        if not run_id:
            raise RuntimeError(f"Camb.ai task completed but no run_id returned: {status}")

        result_resp = requests.get(
            f"{self.base_url}/transcription-result/{run_id}",
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        result_resp.raise_for_status()
        return json.dumps(result_resp.json(), indent=2)


class CambTranslatedTTSTool(_CambToolBase, Tool):
    """Combined translation and text-to-speech tool using the Camb.ai API.

    Translates text from a source language to a target language and synthesizes
    speech from the translated text in one step. Returns the path to the generated audio file.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.
        voice_id (`int`, default `147320`): Numeric ID of the voice to use for speech synthesis.

    Examples:
        ```python
        >>> from smolagents import CambTranslatedTTSTool
        >>> translated_tts = CambTranslatedTTSTool()
        >>> audio_path = translated_tts("Hello!", source_language=1, target_language=2)
        >>> print(audio_path)  # /tmp/...wav
        ```
    """

    name = "camb_translated_tts"
    description = (
        "Translates text and converts it to speech in one step using Camb.ai. "
        "Returns the file path to the generated audio."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to translate and convert to speech.",
        },
        "source_language": {
            "type": "integer",
            "description": "Source language code (integer).",
        },
        "target_language": {
            "type": "integer",
            "description": "Target language code (integer).",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, voice_id: int = 147320, **kwargs):
        self.voice_id = voice_id
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, text: str, source_language: int, target_language: int) -> str:
        import requests

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "source_language": source_language,
            "target_language": target_language,
        }
        resp = requests.post(
            f"{self.base_url}/translated-tts",
            json=payload,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id") or task_data.get("id")
        if not task_id:
            raise RuntimeError(f"Camb.ai API did not return a task ID: {task_data}")

        status = self._poll_task("translated-tts", task_id)
        run_id = status.get("run_id")
        if not run_id:
            raise RuntimeError(f"Camb.ai task completed but no run_id returned: {status}")

        audio_resp = requests.get(
            f"{self.base_url}/tts-result/{run_id}",
            headers={"x-api-key": self.api_key},
            stream=True,
            timeout=self.timeout,
        )
        audio_resp.raise_for_status()
        audio_data = b"".join(audio_resp.iter_content(chunk_size=8192))
        suffix = self._detect_audio_suffix(audio_data)
        return self._save_audio_to_temp(audio_data, suffix=suffix)


class CambVoiceCloneTool(_CambToolBase, Tool):
    """Voice cloning tool that creates a custom voice from an audio sample using the Camb.ai API.

    Creates a new voice profile from a 2+ second audio recording. Returns JSON with the
    created voice ID, which can then be used with `CambTextToSpeechTool` or `CambTranslatedTTSTool`.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambVoiceCloneTool
        >>> cloner = CambVoiceCloneTool()
        >>> result = cloner("my_voice", "sample.wav", gender=1)
        >>> print(result)  # JSON with voice_id
        ```
    """

    name = "camb_voice_clone"
    description = "Clones a voice from a 2+ second audio sample using Camb.ai. Returns JSON with the created voice_id."
    inputs = {
        "voice_name": {
            "type": "string",
            "description": "Name for the cloned voice.",
        },
        "audio_file_path": {
            "type": "string",
            "description": "Path to the audio file (2+ seconds) to clone the voice from.",
        },
        "gender": {
            "type": "integer",
            "description": "Gender code: 0=not specified, 1=male, 2=female, 9=not applicable.",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, voice_name: str, audio_file_path: str, gender: int) -> str:
        import requests

        with open(audio_file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/create-custom-voice",
                headers={"x-api-key": self.api_key},
                files={"file": f},
                data={"voice_name": voice_name, "gender": str(gender)},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


class CambVoiceListTool(_CambToolBase, Tool):
    """Voice listing tool that retrieves all available voices from the Camb.ai API.

    Returns a JSON array of voices with their IDs, names, genders, and languages.
    Useful for discovering voice IDs to use with TTS tools.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambVoiceListTool
        >>> voices = CambVoiceListTool()
        >>> result = voices(language_filter="en-us")
        >>> print(result)  # JSON array of matching voices
        ```
    """

    name = "camb_voice_list"
    description = (
        "Lists all available voices from Camb.ai with ID, name, gender, and language. "
        "Use this to find the right voice_id for TTS tools."
    )
    inputs = {
        "language_filter": {
            "type": "string",
            "description": "Optional language to filter voices by (e.g. 'en-us'). Pass null to list all.",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, language_filter: str | None = None) -> str:
        import requests

        resp = requests.get(
            f"{self.base_url}/list-voices",
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        voices = resp.json()

        voice_list = []
        for voice in voices:
            entry = {
                "id": voice.get("id"),
                "name": voice.get("voice_name", voice.get("name", "Unknown")),
                "gender": self._gender_to_string(voice.get("gender", 0)),
                "age": voice.get("age"),
                "language": voice.get("language"),
            }
            if language_filter and entry.get("language") != language_filter:
                continue
            voice_list.append(entry)

        return json.dumps(voice_list, indent=2)

    @staticmethod
    def _gender_to_string(gender: int) -> str:
        gender_map = {
            0: "not_specified",
            1: "male",
            2: "female",
            9: "not_applicable",
        }
        return gender_map.get(gender, "unknown")


class CambTextToSoundTool(_CambToolBase, Tool):
    """Sound generation tool that creates audio from text descriptions using the Camb.ai API.

    Generates sounds, music, or soundscapes from natural-language prompts.
    Returns the path to the generated audio file.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambTextToSoundTool
        >>> sound_gen = CambTextToSoundTool()
        >>> audio_path = sound_gen("rain falling on a tin roof")
        >>> print(audio_path)  # /tmp/...wav
        ```
    """

    name = "camb_text_to_sound"
    description = (
        "Generates sounds, music, or soundscapes from text descriptions using Camb.ai. "
        "Returns the file path to the generated audio."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Description of the sound or music to generate.",
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, prompt: str) -> str:
        import requests

        payload = {"prompt": prompt}
        resp = requests.post(
            f"{self.base_url}/text-to-sound",
            json=payload,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id") or task_data.get("id")
        if not task_id:
            raise RuntimeError(f"Camb.ai API did not return a task ID: {task_data}")

        status = self._poll_task("text-to-sound", task_id)
        run_id = status.get("run_id")
        if not run_id:
            raise RuntimeError(f"Camb.ai task completed but no run_id returned: {status}")

        audio_resp = requests.get(
            f"{self.base_url}/text-to-sound-result/{run_id}",
            headers={"x-api-key": self.api_key},
            stream=True,
            timeout=self.timeout,
        )
        audio_resp.raise_for_status()
        audio_data = b"".join(audio_resp.iter_content(chunk_size=8192))
        suffix = self._detect_audio_suffix(audio_data)
        return self._save_audio_to_temp(audio_data, suffix=suffix)


class CambAudioSeparationTool(_CambToolBase, Tool):
    """Audio separation tool that splits vocals from background audio using the Camb.ai API.

    Separates an audio file into foreground (vocals/speech) and background tracks.
    Returns JSON with URLs to download the separated audio stems.

    Args:
        api_key (`str`, *optional*): Camb.ai API key. Falls back to the `CAMB_API_KEY` environment variable.

    Examples:
        ```python
        >>> from smolagents import CambAudioSeparationTool
        >>> separator = CambAudioSeparationTool()
        >>> result = separator("song.mp3")
        >>> print(result)  # JSON with vocals and background URLs
        ```
    """

    name = "camb_audio_separation"
    description = (
        "Separates vocals/speech from background audio using Camb.ai. "
        "Returns JSON with URLs for the separated vocals and background tracks."
    )
    inputs = {
        "audio_file_path": {
            "type": "string",
            "description": "Path to the audio file to separate.",
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        self._camb_init(
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k in ("base_url", "timeout", "max_poll_attempts", "poll_interval")},
        )
        super().__init__()

    def forward(self, audio_file_path: str) -> str:
        import requests

        with open(audio_file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/audio-separation",
                headers={"x-api-key": self.api_key},
                files={"media_file": f},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id") or task_data.get("id")
        if not task_id:
            raise RuntimeError(f"Camb.ai API did not return a task ID: {task_data}")

        status = self._poll_task("audio-separation", task_id)
        run_id = status.get("run_id")
        if not run_id:
            raise RuntimeError(f"Camb.ai task completed but no run_id returned: {status}")

        result_resp = requests.get(
            f"{self.base_url}/audio-separation-result/{run_id}",
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        result_resp.raise_for_status()
        result = result_resp.json()

        output = {
            "vocals": result.get("foreground_audio_url"),
            "background": result.get("background_audio_url"),
            "status": "completed",
        }
        return json.dumps(output, indent=2)


__all__ = [
    "CambTextToSpeechTool",
    "CambTranslationTool",
    "CambTranscriptionTool",
    "CambTranslatedTTSTool",
    "CambVoiceCloneTool",
    "CambVoiceListTool",
    "CambTextToSoundTool",
    "CambAudioSeparationTool",
]
