import base64
import io
import os
from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, GoogleSearchTool, LiteLLMModel, Tool, tool
from smolagents.agents import ActionStep

from .omniparser_utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model


github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""  # The agent is able to achieve this request only when powered by GPT-4o or Claude-3.5-sonnet.

search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""


yolo_model = get_yolo_model(model_path='/Users/aymeric/Documents/Code/smolagents/examples/open_deep_research/weights/icon_detect/model.pt')

from copy import deepcopy

def omniparse(
    image_input,
    box_threshold = 0.05,
    iou_threshold = 0.5,
    use_paddleocr = True,
    imgsz = 640,
):
    image_save_path = 'imgs/saved_image_demo.png'
    os.makedirs("imgs", exist_ok=True)
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=None, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    return image, str(parsed_content_list)

class ScreenClicker(Tool):
    name = "click_box"
    description = "Clicks a box on screen."
    inputs = {"box_number": {"type": "integer", "description": "the number on the box on screen to click"}}
    output_type = "null"

    def __init__(self):
        self.coordinates = None
        # self.driver = helium.get_driver()

    def save_screenshot(self, memory_step: ActionStep, agent: CodeAgent) -> None:
        sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
        print("TRIGGERED SCREENSHOT")
        current_step = memory_step.step_number
        self.driver = helium.get_driver()
        if self.driver is not None:
            for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
                if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                    previous_memory_step.observations_images = None
            png_bytes = self.driver.get_screenshot_as_png()
            screenshot = Image.open(BytesIO(png_bytes))
            annotated_screenshot, self.coordinates = omniparse(screenshot)
            print(f"Captured a browser screenshot: {annotated_screenshot.size} pixels")
            memory_step.observations_images = [annotated_screenshot.copy()]  # Create a copy to ensure it persists, important!
            annotated_screenshot.show()
            # Update observations with current URL
            url_info = f"Current url: {self.driver.current_url}"
            memory_step.observations = (
                url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
            )
        return

    def forward(self, box_number: int) -> None:
        x_begin, y_begin, x_end, y_end = self.coordinates[f"icon {box_number}"]
        js_script_click = """
            function clickAtPoint(x, y) {
                const element = document.elementFromPoint(x, y);
                if (element) {
                    const clickEvent = new MouseEvent('click', {
                        view: window,
                        bubbles: true,
                        cancelable: true,
                        clientX: x,
                        clientY: y
                    });
                    element.dispatchEvent(clickEvent);
                }
            }
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            clickAtPoint(
                Math.floor(viewportWidth * arguments[0]), 
                Math.floor(viewportHeight * arguments[1])
            );
        """
        driver.execute_script(js_script_click, (x_begin + x_end) / 2, (y_begin + y_end) / 2)

screen_clicker = ScreenClicker()

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


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[GoogleSearchTool(provider="serper"), go_back, close_popups, search_item_ctrl_f, screen_clicker],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[screen_clicker.save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
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


def make_vision_agent():
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    # Initialize the model based on the provided arguments
    model = LiteLLMModel(
        model_id="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *", agent.state)
    return agent

