# Camb AI Tools

Tools for text-to-speech, translation, transcription, voice cloning, sound generation, and audio separation using the [Camb.ai](https://camb.ai) API.

These tools require a Camb.ai API key, which can be passed directly via the `api_key` parameter or set as the `CAMB_API_KEY` environment variable. All tools depend only on `requests` (a core smolagents dependency).

The Camb AI tools can be grouped by their primary functions:
- **Text-to-Speech**: Convert text into spoken audio.
  - [`CambTextToSpeechTool`]
  - [`CambTranslatedTTSTool`]
- **Translation**: Translate text between 140+ languages.
  - [`CambTranslationTool`]
- **Transcription**: Convert spoken audio to text with speaker detection.
  - [`CambTranscriptionTool`]
- **Voice Cloning & Management**: Clone voices and browse available voices.
  - [`CambVoiceCloneTool`]
  - [`CambVoiceListTool`]
- **Sound Generation**: Generate sounds and music from text prompts.
  - [`CambTextToSoundTool`]
- **Audio Separation**: Split audio into vocal and background stems.
  - [`CambAudioSeparationTool`]

## CambTextToSpeechTool

[[autodoc]] smolagents.camb_tools.CambTextToSpeechTool

## CambTranslationTool

[[autodoc]] smolagents.camb_tools.CambTranslationTool

## CambTranscriptionTool

[[autodoc]] smolagents.camb_tools.CambTranscriptionTool

## CambTranslatedTTSTool

[[autodoc]] smolagents.camb_tools.CambTranslatedTTSTool

## CambVoiceCloneTool

[[autodoc]] smolagents.camb_tools.CambVoiceCloneTool

## CambVoiceListTool

[[autodoc]] smolagents.camb_tools.CambVoiceListTool

## CambTextToSoundTool

[[autodoc]] smolagents.camb_tools.CambTextToSoundTool

## CambAudioSeparationTool

[[autodoc]] smolagents.camb_tools.CambAudioSeparationTool
