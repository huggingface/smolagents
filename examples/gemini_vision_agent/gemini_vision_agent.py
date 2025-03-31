"""
Gemini Vision Agent Example

A simple interactive tool that uses Google's Gemini Vision API to analyze images directly from Hugging Face.

Usage:
  python gemini_vision_agent.py  # Run the interactive image analysis session
"""

import os
import base64
import requests
import time
import logging
import argparse
from functools import wraps
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

try:
    import pyscreenshot as ImageGrab
except ImportError:
    from PIL import ImageGrab

from smolagents import tool, ChatMessage, MessageRole, CodeAgent, LiteLLMModel
from smolagents.monitoring import LogLevel

# Disable info level logging from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm.llms").setLevel(logging.WARNING)
logging.getLogger("litellm.utils").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Configure basic logging for our app
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and setup API key
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")
os.environ["GEMINI_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

# Rate limiter
class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 5):
        self.interval = 60.0 / max_calls_per_minute
        self.last_call_time = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.interval:
            wait_time = self.interval - time_since_last_call
            # Silent waiting
            time.sleep(wait_time)
        
        self.last_call_time = time.time()

# Create global rate limiter
RATE_LIMITER = RateLimiter(max_calls_per_minute=5)

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        RATE_LIMITER.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper

# Define vision tools
@tool
def get_available_images() -> dict:
    """
    Get list of available demo images from Hugging Face.
    
    Returns:
        Dictionary of image names and their URLs.
    """
    return {
        "space.jpeg": "https://huggingface.co/datasets/emredeveloper/images_for_agents/resolve/main/space.jpeg",
        "carbon.png": "https://huggingface.co/datasets/emredeveloper/images_for_agents/resolve/main/carbon.png"
    }

@tool
def display_image(image_name_or_url: str) -> str:
    """
    Display an image from Hugging Face or a direct URL.
    
    Args:
        image_name_or_url: Name of predefined image or a direct URL
        
    Returns:
        Success message or error
    """
    try:
        # Check if it's a predefined image name
        available_images = get_available_images()
        if image_name_or_url in available_images:
            image_url = available_images[image_name_or_url]
        else:
            # Assume it's a URL
            image_url = image_name_or_url
            
        # Download and display the image
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        return f"Successfully displayed image: {image_name_or_url}"
    except Exception as e:
        return f"Error displaying image: {str(e)}"

@tool
@rate_limited
def analyze_image(image_name_or_url: str, prompt: str = "Describe what you see") -> str:
    """
    Analyze an image using Gemini Vision API.
    
    Args:
        image_name_or_url: Name of predefined image or a direct URL
        prompt: Instructions for the analysis
        
    Returns:
        Analysis result from Gemini
    """
    try:
        # Check if it's a predefined image name
        available_images = get_available_images()
        if image_name_or_url in available_images:
            image_url = available_images[image_name_or_url]
        else:
            # Assume it's a URL
            image_url = image_name_or_url
        
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        # Call Gemini API with modified prompt to ensure visual description focus
        visual_prompt = f"Focus only on what you can visually see in this image. {prompt}"
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            visual_prompt, 
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@tool
def extract_code(image_name_or_url: str) -> str:
    """
    Extract code from an image and save it with proper extension.
    
    Args:
        image_name_or_url: Name of predefined image or a direct URL
        
    Returns:
        Extracted code and file path where it was saved
    """
    try:
        # Check if it's a predefined image name
        available_images = get_available_images()
        if image_name_or_url in available_images:
            image_url = available_images[image_name_or_url]
        else:
            # Assume it's a URL
            image_url = image_name_or_url
        
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        # Create prompt to extract code and identify language
        prompt = "Extract any code visible in this image. First, identify the programming language. Then provide ONLY the exact code, with no additional commentary or formatting."
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        # Get the text response
        code_text = response.text
        
        # Determine language and file extension based on code content
        language = "txt"  # Default
        if "def " in code_text or "import " in code_text or "class " in code_text:
            language = "py"
        elif "<html" in code_text.lower() or "<!doctype" in code_text.lower():
            language = "html"
        elif "{" in code_text and ":" in code_text and ";" in code_text:
            language = "css"
        elif "function" in code_text or "var " in code_text or "const " in code_text:
            language = "js"
        
        # Create output directory if it doesn't exist
        output_dir = "generated_code"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the file with language extension
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"code_{timestamp}.{language}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code_text)
        
        # Read file to verify content was written correctly
        with open(filepath, "r", encoding="utf-8") as f:
            saved_content = f.read()
        
        # Return both the code and where it was saved
        return f"Code extracted and saved to {filepath}\n\n{saved_content}"
    except Exception as e:
        return f"Error extracting code: {str(e)}"

@tool
@rate_limited
def compare_images(image1: str, image2: str, comparison_prompt: str) -> str:
    """
    Compare two images using Gemini Vision API.
    
    Args:
        image1: First image name or URL
        image2: Second image name or URL
        comparison_prompt: Instructions for comparison
        
    Returns:
        Comparison results
    """
    try:
        # Get the actual URLs if image names are provided
        available_images = get_available_images()
        
        if image1 in available_images:
            image1_url = available_images[image1]
        else:
            image1_url = image1
            
        if image2 in available_images:
            image2_url = available_images[image2]
        else:
            image2_url = image2
        
        # Download both images
        response1 = requests.get(image1_url)
        response1.raise_for_status()
        image_data1 = base64.b64encode(response1.content).decode('utf-8')
        
        response2 = requests.get(image2_url)
        response2.raise_for_status()
        image_data2 = base64.b64encode(response2.content).decode('utf-8')
        
        # Call Gemini API with both images
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            comparison_prompt,
            {"mime_type": "image/jpeg", "data": image_data1},
            {"mime_type": "image/jpeg", "data": image_data2}
        ])
        return response.text
    except Exception as e:
        return f"Error comparing images: {str(e)}"

@tool
def capture_screenshot() -> str:
    """
    Capture a screenshot of the current screen and save it.
    
    Returns:
        Path to the saved screenshot
    """
    try:
        # Create screenshots directory if it doesn't exist
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Capture the screenshot - handling potential errors with PIL fallback
        try:
            img = ImageGrab.grab()
        except Exception as e:
            # Fallback to PIL's ImageGrab if pyscreenshot fails
            from PIL import ImageGrab as PILImageGrab
            img = PILImageGrab.grab()
        
        # Save screenshot with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
        img.save(filepath)
        
        # Display the screenshot
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        return f"Screenshot saved to {filepath}"
    except Exception as e:
        return f"Error capturing screenshot: {str(e)}"

@tool
def analyze_screenshot(prompt: str = "Describe what you see on this screen") -> str:
    """
    Capture a screenshot and analyze it with Gemini Vision API.
    
    Args:
        prompt: Instructions for the analysis
        
    Returns:
        Analysis result from Gemini
    """
    try:
        # Create screenshots directory if it doesn't exist
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Capture the screenshot
        img = ImageGrab.grab()
        
        # Save screenshot with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
        img.save(filepath)
        
        # Convert to base64 for Gemini API
        with open(filepath, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/png", "data": image_data}
        ])
        
        return f"Screenshot saved to {filepath}\n\nAnalysis:\n{response.text}"
    except Exception as e:
        return f"Error analyzing screenshot: {str(e)}"

def create_smolagent() -> CodeAgent:
    """
    Create a CodeAgent with all tools for structured interaction.
    
    Returns:
        A configured CodeAgent instance
    """
    # Setup the model with more appropriate settings
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        api_key=google_api_key,
        temperature=0.1,  # Reduce temperature for more deterministic behavior
        max_completion_tokens=2048,
        max_retries=2
    )
    
    # Define all tools
    tools = [
        get_available_images,
        display_image,
        analyze_image, 
        extract_code,
        compare_images,
        capture_screenshot,
        analyze_screenshot
    ]
    
    # Define a custom system prompt
    system_prompt = """
    You are a vision analysis assistant that uses tools to help users with image-related tasks.
    
    IMPORTANT RULES:
    1. ONLY use the tools that are directly requested by the user.
    2. Do NOT perform additional analyses or actions unless explicitly asked.
    3. For functions like get_available_images(), just return the result directly.
    4. When asked to display_image, only display the image without analyzing it.
    5. If the user's request is unclear, ask for clarification instead of guessing.
    
    These rules are critical to ensure you perform exactly what the user requests - no more, no less.
    """
    
    # Create the agent with better retry mechanism
    agent = CodeAgent(
        model=model,
        tools=tools,
        name="GeminiVisionAgent",
        description="An agent that can analyze images using Gemini Vision API",
        verbosity_level=LogLevel.INFO,
        max_steps=7,  # Reduce processing steps
        add_base_tools=False,  # Disable basic tools to focus on vision capabilities
        system_prompt=system_prompt  # Add custom system instructions
    )
    
    return agent

def run_agent_session():
    """Run a session with a structured CodeAgent."""
    # Clear the terminal for a clean output
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*70)
    print(" Gemini Vision Agent - CodeAgent Mode")
    print("="*70)
    
    print("\nThis agent uses smolagents CodeAgent architecture to provide:")
    print("* Structured tool usage")
    print("* Better error handling")
    print("* Advanced image analysis")
    print("* Screenshot capture and analysis")
    
    # Create the agent
    agent = create_smolagent()
    
    print("\nAvailable tools:")
    print("- get_available_images() - List available demo images")
    print("- display_image(image) - Display an image")
    print("- analyze_image(image, prompt) - Analyze an image")
    print("- extract_code(image) - Extract code from an image")
    print("- compare_images(image1, image2, prompt) - Compare two images")
    print("- capture_screenshot() - Capture and save a screenshot")
    print("- analyze_screenshot(prompt) - Capture and analyze a screenshot")
    print("\nType 'exit' to quit")
    
    print("Usage examples:")
    print("- \"get_available_images()\" - Just list the available images")
    print("- \"display_image('space.jpeg')\" - Only display the image without analysis")
    print("- \"analyze_image('carbon.png', 'What does this image show?')\" - Analyze with custom prompt")
    print("- \"capture_screenshot()\" - Just take a screenshot without analysis")
    
    # Start interaction loop
    while True:
        user_input = input("\nEnter your request: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        # Make corrections for common user inputs
        if user_input.strip() == "get_available_images":
            user_input = "get_available_images()"
        elif user_input.strip() == "display_image":
            user_input = "display_image('space.jpeg')"
        elif user_input.strip() == "capture_screenshot":
            user_input = "capture_screenshot()"
            
        # Run the agent with the user input
        try:
            response = agent.run(user_input)
            print("\nAgent response:")
            print(response)
        except Exception as e:
            print(f"\nError running agent: {str(e)}")

def run_interactive_session():
    """Run an interactive session allowing the user to directly use tools."""
    # Clear the terminal for a clean output
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*70)
    print(" Gemini Vision Interactive Tool")
    print("="*70)
    
    print("\nCommands:")
    print("1. list              - Show available images")
    print("2. view <image>      - Display an image")
    print("3. analyze <image>   - Analyze image content")
    print("4. code <image>      - Extract & save code from image")
    print("5. compare <img1> <img2> - Compare two images")
    print("6. screenshot        - Capture a screenshot")
    print("7. analyze-screen    - Analyze current screen")
    print("8. agent             - Switch to CodeAgent mode")
    print("9. help              - Show commands")
    print("0. exit              - Quit program")
    
    # Store the available images for quick access
    available_images = get_available_images()
    image_names = list(available_images.keys())
    
    while True:
        try:
            command = input("\n> ").strip()
            parts = command.split()
            
            # Handle empty input
            if not parts:
                continue
                
            cmd = parts[0].lower()
            
            # Simple command parsing
            if cmd in ["exit", "quit", "q"]:
                break
                
            elif cmd in ["help", "?", "h"]:
                print("\nCommands:")
                print("1. list              - Show available images")
                print("2. view <image>      - Display an image")
                print("3. analyze <image>   - Analyze image content")
                print("4. code <image>      - Extract & save code from image")
                print("5. compare <img1> <img2> - Compare two images")
                print("6. screenshot        - Capture a screenshot")
                print("7. analyze-screen    - Analyze current screen")
                print("8. agent             - Switch to CodeAgent mode")
                print("9. help              - Show commands")
                print("0. exit              - Quit program")
                
            elif cmd == "list":
                print("\nAvailable images:")
                for name in image_names:
                    print(f"- {name}")
                    
            elif cmd in ["view", "display", "show"]:
                if len(parts) < 2:
                    print(f"Usage: view <image>\nAvailable: {', '.join(image_names)}")
                    continue
                    
                image_name = parts[1]
                print(f"Displaying {image_name}...")
                result = display_image(image_name)
                print(result)
                
            elif cmd in ["analyze", "describe"]:
                if len(parts) < 2:
                    print(f"Usage: analyze <image>\nAvailable: {', '.join(image_names)}")
                    continue
                    
                image_name = parts[1]
                prompt = " ".join(parts[2:]) if len(parts) > 2 else "Describe what you see in this image"
                print(f"Analyzing {image_name}...")
                result = analyze_image(image_name, prompt)
                print(f"\n{result}")
                
            elif cmd in ["code", "extract"]:
                if len(parts) < 2:
                    print(f"Usage: code <image>\nAvailable: {', '.join(image_names)}")
                    continue
                    
                image_name = parts[1]
                print(f"Extracting code from {image_name}...")
                result = extract_code(image_name)
                print(f"\n{result}")
                
            elif cmd == "compare":
                if len(parts) < 3:
                    print(f"Usage: compare <image1> <image2> [prompt]")
                    print(f"Available: {', '.join(image_names)}")
                    continue
                    
                image1, image2 = parts[1], parts[2]
                prompt = " ".join(parts[3:]) if len(parts) > 3 else "Compare these two images"
                print(f"Comparing {image1} and {image2}...")
                result = compare_images(image1, image2, prompt)
                print(f"\n{result}")
                
            elif cmd in ["screenshot", "capture"]:
                print("Capturing screenshot...")
                result = capture_screenshot()
                print(result)
                
            elif cmd in ["analyze-screen", "screen"]:
                print("Analyzing screen...")
                prompt = " ".join(parts[1:]) if len(parts) > 1 else "Describe what you see on this screen"
                result = analyze_screenshot(prompt)
                print(f"\n{result}")
                
            elif cmd in ["agent", "advanced"]:
                print("Switching to CodeAgent mode...")
                run_agent_session()
                
            else:
                print(f"Unknown command: {cmd}. Type 'help' to see available commands.")
                
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Run the Gemini Vision Interactive Tool."""
    # Ensure API key is set
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY is not set in your environment or .env file.")
        print("Please set this key and try again.")
        return
    
    # Parse command line arguments for more flexibility
    parser = argparse.ArgumentParser(description="Gemini Vision Interactive Tool")
    parser.add_argument("--agent", action="store_true", help="Run in CodeAgent mode")
    parser.add_argument("--screenshot", action="store_true", help="Capture a screenshot immediately")
    args = parser.parse_args()
    
    if args.agent:
        run_agent_session()
    elif args.screenshot:
        print(capture_screenshot())
        analyze = input("Would you like to analyze this screenshot? (y/n): ")
        if analyze.lower().startswith('y'):
            prompt = input("Enter analysis prompt (or press Enter for default): ")
            if not prompt:
                prompt = "Describe what you see on this screen"
            print(analyze_screenshot(prompt))
    else:
        # Run the interactive session
        run_interactive_session()

if __name__ == "__main__":
    # Silence matplotlib warning messages
    plt.set_loglevel('warning')
    main()
