# Gemini Vision Interactive Tool

This interactive tool demonstrates how to use Google's Gemini multimodal model with smolagents for image analysis, code extraction, and image comparisons.

## Features

- Simple command-line interface for image analysis
- Automatic code detection and extraction from images
- Image comparison capabilities
- Built with smolagents and Gemini Vision API

## Requirements

```
pip install -r requirements.txt
```

You'll need a Google API key. Create one at [Google AI Studio](https://ai.google.dev/) and set it as an environment variable:

```
export GOOGLE_API_KEY=your_api_key
```

Or add it to a `.env` file in this directory.

## Running the Tool

```
python gemini_vision_agent.py
```

## Interactive Commands

Once the tool is running, you'll get an interactive prompt where you can use these commands:

- `list` - Show available demo images
- `view <image>` - Display an image  
- `analyze <image>` - Analyze image content
- `code <image>` - Extract and save code from an image
- `compare <img1> <img2>` - Compare two images
- `help` - Show commands
- `exit` - Quit program

## Example Usage

```
> list
> view space.jpeg
> analyze space.jpeg What can you see in this image?
> code carbon.png
> compare space.jpeg carbon.png How do these images differ?
```

## Directory Structure

- `generated_code/` - Contains automatically extracted code files
