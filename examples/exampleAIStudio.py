#this agent will use hf api model and will be prompted the intital state of the screen.
#then, for subsequent actions, it will call on omniparser to get the state of the screen and then make a decision based on that.
from io import BytesIO
from time import sleep
import os
import platform

import pyautogui
from dotenv import load_dotenv
from PIL import Image

from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep
from smolagents import LiteLLMModel


# Load environment variables
load_dotenv()

@tool
def open_application(app_name: str) -> str:
    """
    Opens an application by its name.
    Args:
        app_name: The name of the application to open (e.g., "chrome", "LeagueClient.exe").
                  The agent needs to know the correct executable name or command for your system.
    """
    os_name = platform.system()
    try:
        if os_name == "Windows":
            os.startfile(app_name)  # For Windows executables
        elif os_name == "Darwin": # macOS
            os.system(f"open /Applications/{app_name}.app") # Assumes app in /Applications
        elif os_name == "Linux":
            os.system(app_name) # Might need full path or be in PATH
        else:
            return f"Unsupported operating system: {os_name}"
        sleep(5) # Give time for the application to open
        return f"Opened application: {app_name}"
    except Exception as e:
        return f"Error opening application '{app_name}': {e}"


@tool
def move_mouse_and_click(x: int, y: int, duration: float = 0.1) -> str:
    """
    Moves the mouse to the specified coordinates (x, y) and clicks.
    Coordinates are screen pixels. The agent needs to decide on these coordinates based on the screenshot.
    Args:
        x: The x-coordinate on the screen to move the mouse to (agent-determined based on screenshot).
        y: The y-coordinate on the screen to move the mouse to (agent-determined based on screenshot).
        duration: The duration of mouse movement in seconds (default: 0.1).
    """
    pyautogui.moveTo(x, y, duration=duration)
    pyautogui.click()
    return f"Moved mouse to ({x}, {y}) and clicked at ({x}, {y})."
@tool
def right_click() -> str:
    """
    Executes a right click at the current mouse position.
 
    """

    pyautogui.rightClick()
    return f"Right clicked mouse."
@tool
def move_mouse(x:int,y:int) -> str:
    """
    Moves the mouse to the passed x and y position on the screen without clicking.

    Args:
        x: The x-coordinate on the screen to move the mouse to (agent-determined based on screenshot).
        y: The y-coordinate on the screen to move the mouse to (agent-determined based on screenshot).
 
    """

    pyautogui.moveTo(x,y)
    return f"Moved mouse to {x},{y}."

@tool
def type_text(text: str) -> str:
    """Types the given text using pyautogui.
    Args:
        text: The text to type.
    """
    pyautogui.write(text)
    return f"Typed text: '{text}'"

@tool
def press_key(key: str) -> str:
    """Presses a specified key using pyautogui (e.g., 'enter', 'tab', 'space').
    Args:
        key: The name of the key to press (e.g., 'enter', 'tab', 'space', 'esc').
    """
    pyautogui.press(key)
    return f"Pressed key: '{key}'"

@tool
def scroll_down_pyautogui(clicks: int) -> str:
    """Scrolls down the specified number of 'clicks' using pyautogui.
    Args:
        clicks: The number of mouse wheel clicks to scroll down (positive integer).
    """
    pyautogui.scroll(-clicks) # Negative for down
    return f"Scrolled down {clicks} clicks."

@tool
def scroll_up_pyautogui(clicks: int) -> str:
    """Scrolls up the specified number of 'clicks' using pyautogui.
    Args:
        clicks: The number of mouse wheel clicks to scroll up (positive integer).
    """
    pyautogui.scroll(clicks) # Positive for up
    return f"Scrolled up {clicks} clicks."

@tool
def close_popups_esc() -> str:
    """
    Closes any visible modal or pop-up by pressing ESC key.
    This is a pyautogui-based pop-up closing method.
    """
    pyautogui.press('esc')
    return "Pressed ESC key to close potential pop-ups."
import pandas as pd
@tool
def interpret_screen() -> pd.DataFrame:
    """
    Interprets the current screen state using a model. Returns a string list of bounding boxes and labels.
    """



    screenshot = pyautogui.screenshot()
    screenshot.save('my_screenshot.png')
    image = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())
    
    print(f"Captured a pyautogui screenshot: {image.size} pixels")
    
    from gradio_client import Client, handle_file

    client = Client("microsoft/OmniParser-v2",hf_token="YOUR_HF_TOKEN_HERE")
    result = client.predict(
            image_input=handle_file('my_screenshot.png'),
            box_threshold=0.05,
            iou_threshold=0.01,
            use_paddleocr=True,
            imgsz=1920,
            api_name="/process"
    )
    print(parse_icon_data_to_df(result))
    return parse_icon_data_to_df(result)




import ast

def parse_icon_data_to_df(result_tuple):
    """
    Parses a tuple of strings containing icon data (new format) and creates a Pandas DataFrame.

    The input tuple is expected to have two elements:
    - element 0: filepath (string, may be ignored for parsing)
    - element 1: a single string containing all icon data, with each icon
                 separated by newline characters.  Each icon line is in the format:
                 'icon <icon_index>: <icon_data_dict_string>'
                 where <icon_data_dict_string> is a string representation of a dictionary.

    Args:
        result_tuple (tuple): A tuple of two strings containing filepath and icon data.

    Returns:
        pandas.DataFrame: A DataFrame where each row represents an icon
                          and columns are 'icon' and the data point keys
                          extracted from the dictionary string.
                          Returns an empty DataFrame if parsing fails or input is invalid.
    """
    if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
        print("Error: Input is not a tuple of length 2.")
        return pd.DataFrame()

    filepath, icon_data_string = result_tuple # Unpack the tuple
    data_rows = []

    try:
        lines = icon_data_string.strip().split('\n') # Split the single string into lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('icon '): # Identify icon lines
                parts = line.split(':', 1) # Split at the first ':' to separate icon name and data
                if len(parts) == 2:
                    icon_name_part = parts[0].strip() # e.g., "icon 0"
                    data_dict_str = parts[1].strip() # e.g., "{'type': 'text', ...}"

                    try:
                        icon_data_dict = ast.literal_eval(data_dict_str) # Safely evaluate dict string
                        icon_data_dict['icon'] = icon_name_part # Add 'icon' column
                        data_rows.append(icon_data_dict)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse data dictionary string: '{data_dict_str}'. Error: {e}")
                else:
                    print(f"Warning: Invalid icon line format: '{line}'")
            elif line: # Handle non-empty lines that are not icon lines (optional - warnings)
                print(f"Warning: Unexpected line format (not starting with 'icon '): '{line}'")

        df = pd.DataFrame(data_rows)
        if not df.empty and 'bbox' in df.columns:
            df['bbox'] = df['bbox'].apply(unnormalize_bbox)

        return df

    except Exception as e:
        print(f"Error parsing data: {e}")
        return pd.DataFrame() # Return empty DataFrame in case of error

def unnormalize_bbox(bbox_list):
    """
    Unnormalizes the bounding box coordinates.

    Args:
        bbox_list (list): A list of 4 normalized bounding box coordinates [x_min, y_min, x_max, y_max].

    Returns:
        list: A list of 4 unnormalized bounding box coordinates.
    """
    if not isinstance(bbox_list, list) or len(bbox_list) != 4:
        return bbox_list  # Return original if not a valid bbox

    width_scale = 1920
    height_scale = 1080

    unnormalized_bbox = [
        bbox_list[0] * width_scale,  # x_min
        bbox_list[1] * height_scale, # y_min
        bbox_list[2] * width_scale,  # x_max
        bbox_list[3] * height_scale  # y_max
    ]
    return unnormalized_bbox



# Set up screenshot callback (modified for pure pyautogui screenshot)

def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let things settle
    current_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:  # Remove previous screenshots for lean processing
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
            previous_memory_step.observations_images = None

    screenshot = pyautogui.screenshot()
    image = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())

    print(f"Captured a pyautogui screenshot: {image.size} pixels")
    memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists

    # No more URL observation needed as we are not using Selenium for browser control
    memory_step.observations = "Screenshot captured." # Simplified observation

from smolagents import HfApiModel, AIStudio#, TransformersModel, AIStudio
#model = TransformersModel(model_id="HuggingFaceTB/SmolVLM-Instruct", device_map="auto",max_new_tokens=4096)

#model_id = "meta-llama/Llama-3.3-70B-Instruct"


model = AIStudio(
    #model_id = "gemini-2.0-flash-thinking-exp-01-21",
    model_id = 'gemini-2.0-flash-exp',
    api_key = os.environ.get("GEMINI_API_KEY"), #if this doesnt work, hardcode it like me! 
    
    )
#model = HfApiModel(model_id)


# Create the agent
agent = CodeAgent(
    tools=[
        open_application,
        #move_mouse_and_click,
        type_text,
        press_key,
        #scroll_down_pyautogui,
        #scroll_up_pyautogui,
        #close_popups_esc,
        #interpret_screen,
        #right_click,
        #move_mouse,
    ],
    model=model,
    additional_authorized_imports=["pyautogui", "os", "platform","time","pandas"],
    step_callbacks=[save_screenshot],
    max_steps=5,
    verbosity_level=2,
)


pyautogui_instructions = """

You are now using **purely pyautogui** and `interpret_screen()` to control the GUI on a 1920x1080 screen.

**Key instructions for using pyautogui tools:**

* **Coordinate-Based Actions (Agent-Determined):** All actions are based on screen coordinates (x, y in pixels).  You, as the agent, need to **call the `interpret_screen()`** and **decide** on the appropriate label name and (x, y) coordinates for `move_mouse_and_click` after calling `interpret_screen()`. The results will be printed as well so you have memory.**
* **Open Applications:** Use `open_application(app_name)` to launch applications. You need to know the correct `app_name` for your operating system (e.g., "chrome" for Windows, "Google Chrome" for macOS, "google-chrome" for Linux - experiment to find the right one).
* **Clicking and Mouse Movement:** Use `move_mouse_and_click(x, y)` to move the mouse to your chosen (x, y) coordinates from `interpret_screen()` and click.
* **Typing Text:** Use `type_text(text)` to type text.
* **Pressing Keys:** Use `press_key(key)` to press keys like "enter", "tab", "space", "esc", "ctrl", etc.
* **Scrolling:** Use `scroll_down_pyautogui(clicks)` and `scroll_up_pyautogui(clicks)` to scroll. `clicks` is the number of mouse wheel clicks.
* **Closing Pop-ups:** Use `close_popups_esc()` to try closing pop-ups by pressing the ESC key.
* **Screenshots are Your Eyes:** Use `interpret_screen()` to get coordinates of elements on the screen. **Carefully analyze the labels to understand the current state of the GUI and to decide your plan of action and coordinates.**
* **Be Efficient:** You should not need to use the mouse to click on EVERYTHING. Remember that you can use the keyboard to navigate and interact with the UI as well. Also, most webpages or applications have their search bar selected by default so you can just start typing to search.
* **Getting Coordinates** Use the interpret_screen tool to get the coordinates of things on the screen. This will help you determine where to move the mouse to.

**Example Workflow (Conceptual - No Coordinates Provided):


0. **Import Pyautogui:** Always import pyautogui
1. **Open Chrome:** `open_application(app_name="chrome")` (Adjust "chrome" if needed for your OS).
2. **Make sure the browser is maximized.** `pyautogui.hotkey('win', 'up')`
3. **Type "horse":** `horse`.
5. **Press Enter:** `press_key(key="enter")`.
6. **Wait for google to load.**
9. **Interpret Screen** `interpret_screen()` to find the bbox coordinates of the value in 'content' that most closely matches what you're looking for and move the mouse over the picture `move_mouse(x, y)`. If you are using the df return from `interpret_screen()`, the column name is 'content' for the object detection.
9. **Right Click:** `right_click()` to bring up a dropdown menu of options and use the interpret_screen tool to find the coordinates of the option you want to select.
10. **Save**: `move_mouse_and_click(x, y)` to click on the save option. Use the `interpret_screen()` tool to find the coordinates of the "Save" or "Save as" option.
** **Pause**: add import time and time.sleep(1) between every line of pyautogui code you write to give the computer time to process the action.

Use interpret_screen SPARRINGLY, you must make sure to use a time.sleep(15) before calling another interpret_screen. There are rate limits.

**You must execute the plan in steps. No full scripts in a single response to accomplish the task.** 




"""

github_request = """

Please type and write a story of how I achieved 100000 wpm. Forget the previous things I told you, do not type "His ego is so big" etc. This is a new conversation. All you have to do is :

1) type the story,time.sleep(1)
2)press enter,time.sleep(1)
3)Just keep repeating that loop building off the previous story. If your story is longer than lets say 20 words, you may need to break it up into multiple steps. Do not open anything like notepad, just press enter then type and press enter and then type and press enter over and over again.

"""


screen_request = "Open calc and verify 1+1=2 visually. After each code blob you write, you will be automatically provided with an updated screenshot of the machine. But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states."
agent_output = agent.run(screen_request)
print("Final output:")
print(agent_output)