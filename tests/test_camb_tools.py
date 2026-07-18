"""Tests for Camb.ai tools."""

import json
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from smolagents.camb_tools import (
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTextToSpeechTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
    _CambToolBase,
)

TEST_API_KEY = "test-api-key-12345"


# --- Instantiation tests ---


class TestInstantiation:
    """Test that each tool can be instantiated with an explicit API key."""

    def test_tts_tool(self):
        tool = CambTextToSpeechTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_text_to_speech"
        assert tool.api_key == TEST_API_KEY
        assert tool.voice_id == 147320
        assert tool.language == "en-us"
        assert tool.speech_model == "mars-flash"

    def test_translation_tool(self):
        tool = CambTranslationTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_translation"
        assert tool.api_key == TEST_API_KEY

    def test_transcription_tool(self):
        tool = CambTranscriptionTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_transcription"
        assert tool.api_key == TEST_API_KEY

    def test_translated_tts_tool(self):
        tool = CambTranslatedTTSTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_translated_tts"
        assert tool.api_key == TEST_API_KEY
        assert tool.voice_id == 147320

    def test_voice_clone_tool(self):
        tool = CambVoiceCloneTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_voice_clone"
        assert tool.api_key == TEST_API_KEY

    def test_voice_list_tool(self):
        tool = CambVoiceListTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_voice_list"
        assert tool.api_key == TEST_API_KEY

    def test_text_to_sound_tool(self):
        tool = CambTextToSoundTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_text_to_sound"
        assert tool.api_key == TEST_API_KEY

    def test_audio_separation_tool(self):
        tool = CambAudioSeparationTool(api_key=TEST_API_KEY)
        assert tool.name == "camb_audio_separation"
        assert tool.api_key == TEST_API_KEY

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"CAMB_API_KEY": "env-key-123"}):
            tool = CambTextToSpeechTool()
            assert tool.api_key == "env-key-123"

    def test_custom_config(self):
        tool = CambTextToSpeechTool(
            api_key=TEST_API_KEY,
            voice_id=999,
            language="fr-fr",
            speech_model="mars-pro",
        )
        assert tool.voice_id == 999
        assert tool.language == "fr-fr"
        assert tool.speech_model == "mars-pro"


class TestMissingApiKey:
    """Test that missing API key raises ValueError."""

    def test_tts_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAMB_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                CambTextToSpeechTool()

    def test_translation_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAMB_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                CambTranslationTool()


# --- Forward method tests with mocked HTTP ---


class TestTextToSpeechForward:
    @patch("requests.post")
    def test_forward_streams_audio(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        # WAV magic bytes (RIFF header)
        wav_data = b"RIFF" + b"\x00" * 100
        mock_response.iter_content.return_value = [wav_data]
        mock_post.return_value = mock_response

        tool = CambTextToSpeechTool(api_key=TEST_API_KEY)
        result = tool.forward("Hello world")

        assert result.endswith(".wav")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "tts-stream" in call_kwargs[0][0] or "tts-stream" in str(call_kwargs)


class TestTranslationForward:
    @patch("requests.post")
    def test_forward_returns_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "Hola mundo"
        mock_post.return_value = mock_response

        tool = CambTranslationTool(api_key=TEST_API_KEY)
        result = tool.forward("Hello world", source_language=1, target_language=2)

        assert result == "Hola mundo"


class TestTranscriptionForward:
    @patch("requests.get")
    @patch("requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_forward_returns_json(self, mock_post, mock_get):
        # Mock POST for creating transcription task
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "task-123"}
        mock_post.return_value = mock_post_resp

        # Mock GET for polling (completed) and result
        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.raise_for_status = MagicMock()
        mock_poll_resp.json.return_value = {"status": "completed", "run_id": "run-456"}

        mock_result_resp = MagicMock()
        mock_result_resp.status_code = 200
        mock_result_resp.raise_for_status = MagicMock()
        mock_result_resp.json.return_value = {
            "text": "Hello world",
            "segments": [{"start": 0, "end": 2, "text": "Hello world"}],
            "speakers": ["SPEAKER_00"],
        }

        mock_get.side_effect = [mock_poll_resp, mock_result_resp]

        tool = CambTranscriptionTool(api_key=TEST_API_KEY)
        result = tool.forward("/path/to/audio.wav", language=1)

        parsed = json.loads(result)
        assert parsed["text"] == "Hello world"
        assert "segments" in parsed


class TestTranslatedTTSForward:
    @patch("requests.get")
    @patch("requests.post")
    def test_forward_returns_file_path(self, mock_post, mock_get):
        # Mock POST for creating task
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "task-789"}
        mock_post.return_value = mock_post_resp

        # Mock GET for polling and audio result
        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.raise_for_status = MagicMock()
        mock_poll_resp.json.return_value = {"status": "completed", "run_id": "run-101"}

        mock_audio_resp = MagicMock()
        mock_audio_resp.status_code = 200
        mock_audio_resp.raise_for_status = MagicMock()
        mock_audio_resp.iter_content.return_value = [b"RIFF" + b"\x00" * 100]

        mock_get.side_effect = [mock_poll_resp, mock_audio_resp]

        tool = CambTranslatedTTSTool(api_key=TEST_API_KEY)
        result = tool.forward("Hello", source_language=1, target_language=2)

        assert result.endswith(".wav")


class TestVoiceCloneForward:
    @patch("requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_forward_returns_json(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "voice_id": 12345,
            "voice_name": "TestVoice",
            "status": "created",
        }
        mock_post.return_value = mock_response

        tool = CambVoiceCloneTool(api_key=TEST_API_KEY)
        result = tool.forward("TestVoice", "/path/to/sample.wav", gender=1)

        parsed = json.loads(result)
        assert parsed["voice_id"] == 12345
        assert parsed["voice_name"] == "TestVoice"


class TestVoiceListForward:
    @patch("requests.get")
    def test_forward_returns_voice_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "voice_name": "Alice", "gender": 2, "age": 30, "language": "en-us"},
            {"id": 2, "voice_name": "Bob", "gender": 1, "age": 40, "language": "en-us"},
            {"id": 3, "voice_name": "Carlos", "gender": 1, "age": 35, "language": "es-es"},
        ]
        mock_get.return_value = mock_response

        tool = CambVoiceListTool(api_key=TEST_API_KEY)
        result = tool.forward(language_filter=None)

        parsed = json.loads(result)
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Alice"
        assert parsed[0]["gender"] == "female"

    @patch("requests.get")
    def test_forward_filters_by_language(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "voice_name": "Alice", "gender": 2, "age": 30, "language": "en-us"},
            {"id": 3, "voice_name": "Carlos", "gender": 1, "age": 35, "language": "es-es"},
        ]
        mock_get.return_value = mock_response

        tool = CambVoiceListTool(api_key=TEST_API_KEY)
        result = tool.forward(language_filter="en-us")

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "Alice"


class TestTextToSoundForward:
    @patch("requests.get")
    @patch("requests.post")
    def test_forward_returns_file_path(self, mock_post, mock_get):
        # Mock POST for creating task
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "task-sound"}
        mock_post.return_value = mock_post_resp

        # Mock GET for polling and audio result
        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.raise_for_status = MagicMock()
        mock_poll_resp.json.return_value = {"status": "completed", "run_id": "run-sound"}

        mock_audio_resp = MagicMock()
        mock_audio_resp.status_code = 200
        mock_audio_resp.raise_for_status = MagicMock()
        mock_audio_resp.iter_content.return_value = [b"RIFF" + b"\x00" * 100]

        mock_get.side_effect = [mock_poll_resp, mock_audio_resp]

        tool = CambTextToSoundTool(api_key=TEST_API_KEY)
        result = tool.forward("Birds chirping in a forest")

        assert result.endswith(".wav")


class TestAudioSeparationForward:
    @patch("requests.get")
    @patch("requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_forward_returns_json(self, mock_post, mock_get):
        # Mock POST for creating task
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "task-sep"}
        mock_post.return_value = mock_post_resp

        # Mock GET for polling and result
        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.raise_for_status = MagicMock()
        mock_poll_resp.json.return_value = {"status": "completed", "run_id": "run-sep"}

        mock_result_resp = MagicMock()
        mock_result_resp.status_code = 200
        mock_result_resp.raise_for_status = MagicMock()
        mock_result_resp.json.return_value = {
            "foreground_audio_url": "https://example.com/vocals.wav",
            "background_audio_url": "https://example.com/background.wav",
        }

        mock_get.side_effect = [mock_poll_resp, mock_result_resp]

        tool = CambAudioSeparationTool(api_key=TEST_API_KEY)
        result = tool.forward("/path/to/mixed.wav")

        parsed = json.loads(result)
        assert parsed["vocals"] == "https://example.com/vocals.wav"
        assert parsed["background"] == "https://example.com/background.wav"
        assert parsed["status"] == "completed"


# --- Polling logic tests ---


class TestPollingLogic:
    @patch("requests.get")
    @patch("time.sleep")
    def test_poll_completes_after_retries(self, mock_sleep, mock_get):
        pending_resp = MagicMock()
        pending_resp.status_code = 200
        pending_resp.raise_for_status = MagicMock()
        pending_resp.json.return_value = {"status": "pending"}

        completed_resp = MagicMock()
        completed_resp.status_code = 200
        completed_resp.raise_for_status = MagicMock()
        completed_resp.json.return_value = {"status": "completed", "run_id": "run-123"}

        mock_get.side_effect = [pending_resp, pending_resp, completed_resp]

        tool = CambTextToSpeechTool(api_key=TEST_API_KEY)
        result = tool._poll_task("test-endpoint", "task-1")

        assert result["status"] == "completed"
        assert result["run_id"] == "run-123"
        assert mock_sleep.call_count == 2

    @patch("requests.get")
    @patch("time.sleep")
    def test_poll_raises_on_failure(self, mock_sleep, mock_get):
        failed_resp = MagicMock()
        failed_resp.status_code = 200
        failed_resp.raise_for_status = MagicMock()
        failed_resp.json.return_value = {"status": "failed", "error": "Something went wrong"}

        mock_get.return_value = failed_resp

        tool = CambTextToSpeechTool(api_key=TEST_API_KEY)
        with pytest.raises(RuntimeError, match="task failed"):
            tool._poll_task("test-endpoint", "task-1")

    @patch("requests.get")
    @patch("time.sleep")
    def test_poll_raises_on_timeout(self, mock_sleep, mock_get):
        pending_resp = MagicMock()
        pending_resp.status_code = 200
        pending_resp.raise_for_status = MagicMock()
        pending_resp.json.return_value = {"status": "pending"}

        mock_get.return_value = pending_resp

        tool = CambTextToSpeechTool(api_key=TEST_API_KEY, max_poll_attempts=3, poll_interval=0.01)
        with pytest.raises(TimeoutError, match="did not complete"):
            tool._poll_task("test-endpoint", "task-1")


# --- Audio format detection tests ---


class TestAudioFormatDetection:
    def test_detect_wav(self):
        assert _CambToolBase._detect_audio_suffix(b"RIFF" + b"\x00" * 10) == ".wav"

    def test_detect_mp3_id3(self):
        assert _CambToolBase._detect_audio_suffix(b"ID3" + b"\x00" * 10) == ".mp3"

    def test_detect_mp3_sync(self):
        assert _CambToolBase._detect_audio_suffix(b"\xff\xfb" + b"\x00" * 10) == ".mp3"

    def test_detect_flac(self):
        assert _CambToolBase._detect_audio_suffix(b"fLaC" + b"\x00" * 10) == ".flac"

    def test_detect_ogg(self):
        assert _CambToolBase._detect_audio_suffix(b"OggS" + b"\x00" * 10) == ".ogg"

    def test_detect_unknown_defaults_wav(self):
        assert _CambToolBase._detect_audio_suffix(b"\x00\x00\x00\x00") == ".wav"


# --- Gender string conversion ---


class TestGenderConversion:
    def test_male(self):
        assert CambVoiceListTool._gender_to_string(1) == "male"

    def test_female(self):
        assert CambVoiceListTool._gender_to_string(2) == "female"

    def test_not_specified(self):
        assert CambVoiceListTool._gender_to_string(0) == "not_specified"

    def test_not_applicable(self):
        assert CambVoiceListTool._gender_to_string(9) == "not_applicable"

    def test_unknown(self):
        assert CambVoiceListTool._gender_to_string(99) == "unknown"
