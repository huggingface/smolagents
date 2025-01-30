import argparse
import os
from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, OpenAIServerModel, TransformersModel, tool  # noqa: F401
from smolagents.agents import ActionStep


github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""  # The agent is able to achieve this request only when powered by GPT-4o or Claude-3.5-sonnet.

search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "--model",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, HfApiModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument("--prompt", type=str, default=search_request, help="The prompt to run with the agent")
    return parser.parse_args()


# Load environment variables
load_dotenv()

# Parse command line arguments
args = parse_arguments()

# Initialize the model based on the provided arguments
if args.model == "OpenAIServerModel":
    model = OpenAIServerModel(
        api_key=os.getenv("FIREWORKS_API_KEY"),
        api_base="https://api.fireworks.ai/inference/v1",
        model_id=args.model_id,
    )
elif args.model == "LiteLLMModel":
    model = LiteLLMModel(
        model_id=args.model_id,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif args.model == "TransformersModel":
    model = TransformersModel(model_id=args.model_id, device_map="auto", flatten_messages_as_text=False)
elif args.model == "HfApiModel":
    model = HfApiModel(
        token=os.getenv("HF_API_KEY"),
        model_id=args.model_id,
    )
else:
    raise ValueError(f"Unsupported model type: {args.model}")


# Prepare callback
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


# Initialize driver and agent
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--window-size=1000,1350")
chrome_options.add_argument("--disable-pdf-viewer")
chrome_options.add_argument("--window-position=0,0")

driver = helium.start_chrome(headless=False, options=chrome_options)

# Initialize tools


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


agent = CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""

# Run the agent with the provided prompt

agent.python_executor("from helium import *", agent.state)
agent.run(args.prompt + helium_instructions)
