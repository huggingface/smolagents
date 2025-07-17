# 📚 管理Agent的记忆

[[open-in-colab]]

归根结底，Agent可以定义为由几个简单组件构成：它拥有工具、提示词。最重要的是，它具备对过往步骤的记忆，能够追溯完整的规划、执行和错误历史。

### 回放Agent的记忆

我们提供了多项功能来审查Agent的过往运行记录。

您可以通过插装（instrumentation）在可视化界面中查看Agent的运行过程，该界面支持对特定步骤进行缩放操作，具体方法参见[插装指南](./inspect_runs)。

您也可以使用`agent.replay()`方法实现回放：

当Agent完成运行后：
```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=0)

result = agent.run("What's the 20th Fibonacci number?")
```

若要回放最近一次运行，只需使用：
```py
agent.replay()
```

### 动态修改Agent的记忆

许多高级应用场景需要对Agent的记忆进行动态修改。

您可以通过以下方式访问Agent的记忆：

```py
from smolagents import ActionStep

system_prompt_step = agent.memory.system_prompt
print("The system prompt given to the agent was:")
print(system_prompt_step.system_prompt)

task_step = agent.memory.steps[0]
print("\n\nThe first task step was:")
print(task_step.task)

for step in agent.memory.steps:
    if isinstance(step, ActionStep):
        if step.error is not None:
            print(f"\nStep {step.step_number} got this error:\n{step.error}\n")
        else:
            print(f"\nStep {step.step_number} got these observations:\n{step.observations}\n")
```

使用`agent.memory.get_full_steps()`可获取完整步骤字典数据。

您还可以通过步骤回调（step callbacks）实现记忆的动态修改。

步骤回调函数可通过参数直接访问`agent`对象，因此能够访问所有记忆步骤并根据需要进行修改。例如，假设您正在监控网页浏览Agent每个步骤的屏幕截图，希望保留最新截图同时删除旧步骤的图片以节省token消耗。

可参考以下代码示例：
_注：此代码片段不完整，部分导入语句和对象定义已精简，完整代码请访问[原始脚本](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py)_

```py
import helium
from PIL import Image
from io import BytesIO
from time import sleep

def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= latest_step - 2:
            previous_memory_step.observations_images = None
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]
```

最后在初始化Agent时，将此函数传入`step_callbacks`参数：

```py
CodeAgent(
    tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[update_screenshot],
    max_steps=20,
    verbosity_level=2,
)
```

请访问我们的 [vision web browser code](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py) 查看完整可运行示例。

### 分步运行 Agents

当您需要处理耗时数天的工具调用时，这种方式特别有用：您可以逐步执行Agents。这还允许您在每一步更新记忆。

```py
from smolagents import InferenceClientModel, CodeAgent, ActionStep, TaskStep

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=1)
print(agent.memory.system_prompt)

task = "What is the 20th Fibonacci number?"

# You could modify the memory as needed here by inputting the memory of another agent.
# agent.memory.steps = previous_agent.memory.steps

# Let's start a new task!
agent.memory.steps.append(TaskStep(task=task, task_images=[]))

final_answer = None
step_number = 1
while final_answer is None and step_number <= 10:
    memory_step = ActionStep(
        step_number=step_number,
        observations_images=[],
    )
    # Run one step.
    final_answer = agent.step(memory_step)
    agent.memory.steps.append(memory_step)
    step_number += 1

    # Change the memory as you please!
    # For instance to update the latest step:
    # agent.memory.steps[-1] = ...

print("The final answer is:", final_answer)
```