"""
Gemini Vision Agent Example

A simple example showing how to use Gemini with smolagents for image analysis.

Usage:
  python gemini_vision_agent.py --help
  python gemini_vision_agent.py --ui      # Start the Gradio UI
  python gemini_vision_agent.py --sample  # Run a sample analysis
"""

import os
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    GradioUI,
    tool
)

# Load environment variables and setup Google API
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Check if API key is available
if not google_api_key:
    print("\n" + "="*80)
    print("ERROR: Google API key not found!")
    print("="*80)
    print("You need to set your Google API key to use this example.")
    print("\nOptions to set your API key:")
    print("1. Create a .env file in this directory with the line: GOOGLE_API_KEY=your_key_here")
    print("2. Set an environment variable: export GOOGLE_API_KEY=your_key_here")
    print("\nTo get an API key:")
    print("1. Go to https://ai.google.dev/")
    print("2. Create an account or sign in")
    print("3. Create an API key in Google AI Studio")
    print("="*80 + "\n")
    exit(1)

# Set the API key for LiteLLM and Google AI
os.environ["GEMINI_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

# Define vision tools
@tool
def list_images() -> str:
    """
    List all image files in the 'images' directory.
    
    Returns:
        A string with newline-separated image paths or an error message.
    """
    images_dir = Path("images")
    if not images_dir.exists():
        return "Images directory not found."
    
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([str(p) for p in images_dir.glob(f"*{ext}")])
    
    return "\n".join(image_files) if image_files else "No images found."

@tool
def get_space_image_path() -> str:
    """
    Get the path to the space.jpeg image.
    
    Returns:
        A string with the path to space.jpeg or an error message if not found.
    """
    image_path = Path("images/space.jpeg")
    return str(image_path) if image_path.exists() else "Space image not found."

@tool
def analyze_image(image_path: str, prompt: str) -> str:
    """
    Analyze an image using Gemini Vision API with a specific prompt.
    
    Args:
        image_path: Path to the image file to analyze
        prompt: Text prompt describing what to analyze in the image
        
    Returns:
        The analysis result from Gemini or an error message.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@tool
def save_text(content: str, filename: str = "analysis_result.txt") -> str:
    """
    Save text content to a file.
    
    Args:
        content: The text content to save
        filename: Name of the file to save to (default: "analysis_result.txt")
        
    Returns:
        A success message or an error message.
    """
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Content saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def display_image(image_path: str) -> str:
    """
    Display an image using matplotlib.
    
    Args:
        image_path: Path to the image file to display
        
    Returns:
        A success message or an error message.
    """
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return "Image displayed successfully"
    except Exception as e:
        return f"Error displaying image: {str(e)}"

def create_agent():
    """Create and configure a Gemini Vision agent."""
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        temperature=0.1,
        max_completion_tokens=4096
    )
    
    return CodeAgent(
        model=model,
        tools=[list_images, get_space_image_path, analyze_image, save_text, display_image],
        name="Gemini Vision Agent",
        description="An agent that can analyze images using Google's Gemini model"
    )

def main():
    """Process command line arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="Gemini Vision Agent")
    parser.add_argument('--ui', action='store_true', help='Launch the Gradio UI')
    parser.add_argument('--sample', action='store_true', help='Run a sample analysis')
    
    args = parser.parse_args()
    
    if args.ui:
        # Launch Gradio UI
        agent = create_agent()
        GradioUI(agent).launch()
    elif args.sample:
        # Run sample analysis
        agent = create_agent()
        prompt = """
        Here's what to do:
        1. Get the path to the space.jpeg image
        2. Display the image
        3. Analyze the image with this prompt: "Describe this space image in detail"
        4. Save the results to "space_analysis.txt"
        """
        result = agent.run(prompt)
        print("\nAnalysis Result:")
        print(result)
    else:
        # Show help information
        print("Please specify an option:")
        print("  --ui      Launch the Gradio UI")
        print("  --sample  Run a sample image analysis")
        print("  --help    Show this help message")
        print("\nExample: python gemini_vision_agent.py --ui")

if __name__ == "__main__":
    main()
