# Gemini Vision Agent

This simple example demonstrates how to use Google's Gemini multimodal model with smolagents for image analysis.

## Features

- Implements a minimal set of vision-related tools
- Shows how to use Gemini's image understanding capabilities
- Provides both UI and programmatic interfaces

## Requirements

```
pip install -e ../.. google-generativeai pillow matplotlib
```

You'll also need a Google API key. Create one at [Google AI Studio](https://ai.google.dev/) and set it as an environment variable:

```
export GOOGLE_API_KEY=your_api_key
```

Or add it to a `.env` file in this directory.

## Running the Example

```
python gemini_vision_agent.py
```

To run the sample usage, run:

```
python gemini_vision_agent.py --sample
```
