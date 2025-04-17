from io import BytesIO

import requests
from PIL import Image

from smolagents import CodeAgent, GradioUI, InferenceClientModel


def add_agent_image(memory_step, agent):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/smolagents.png"
    response = requests.get(url)
    memory_step.observations_images = [Image.open(BytesIO(response.content))]


agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent that has not tool but will always see an agent at the end of its step.",
    step_callbacks=[add_agent_image],
)

GradioUI(agent, file_upload_folder="./data").launch()
