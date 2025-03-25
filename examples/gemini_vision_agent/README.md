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

To set up the sample directory, run:

```
python gemini_vision_agent.py --setup
```

To start the UI, run:

```
python gemini_vision_agent.py --ui
```

or simply:

```
python gemini_vision_agent.py
```

To run the sample usage, run:

```
python gemini_vision_agent.py --sample
```

This will start an interactive session with the Gemini Vision agent where you can upload images and ask questions about them.

## Files that can be deleted

1. `sample_usage.py` - all functionality has been moved to the main file
2. `setup_sample.py` - all functionality has been moved to the main file
