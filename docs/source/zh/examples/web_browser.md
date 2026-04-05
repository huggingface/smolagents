# 使用Agent实现网页浏览器自动化 🤖🌐

[[open-in-colab]]

在本notebook中，我们将创建一个**基于Agent的网页浏览器自动化系统**！该系统可以自动导航网站、与网页元素交互并提取信息。

该Agent将能够：

- [x] 导航到网页
- [x] 点击元素
- [x] 在页面内搜索
- [x] 处理弹出窗口和模态框
- [x] 提取信息

让我们一步步搭建这个系统！

首先运行以下命令安装所需依赖：

```bash
pip install smolagents selenium helium pillow -q
```

让我们导入所需的库并设置环境变量：

```python
from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep

# Load environment variables
load_dotenv()
```

现在我们来创建核心的浏览器交互工具，使我们的Agent能够导航并与网页交互：

```python
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
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
```

让我们配置使用Chrome浏览器并设置截图功能：

```python
# Configure Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--window-size=1000,1350")
chrome_options.add_argument("--disable-pdf-viewer")
chrome_options.add_argument("--window-position=0,0")

# Initialize the browser
driver = helium.start_chrome(headless=False, options=chrome_options)

# Set up screenshot callback
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
```

现在我们来创建网页自动化Agent：

```python
from smolagents import InferenceClientModel

# Initialize the model
model_id = "meta-llama/Llama-3.3-70B-Instruct"  # You can change this to your preferred model
model = InferenceClientModel(model_id=model_id)

# Create the agent
agent = CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

# Import helium for the agent
agent.python_executor("from helium import *", agent.state)
```

Agent需要获得关于如何使用Helium进行网页自动化的指导。以下是我们将提供的操作说明：

```python
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
"""
```

现在我们可以运行Agent执行任务了！让我们尝试在维基百科上查找信息：

```python
search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""

agent_output = agent.run(search_request + helium_instructions)
print("Final output:")
print(agent_output)
```

您可以通过修改请求参数执行不同任务。例如，以下请求可帮助我判断是否需要更加努力工作：

```python
github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""

agent_output = agent.run(github_request + helium_instructions)
print("Final output:")
print(agent_output)
```

该系统在以下任务中尤为有效：

- 从网站提取数据
- 网页研究自动化
- 用户界面测试与验证
- 内容监控