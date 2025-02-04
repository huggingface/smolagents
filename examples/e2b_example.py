from dotenv import load_dotenv

from smolagents import CodeAgent, HfApiModel, Tool
from smolagents.default_tools import VisitWebpageTool


load_dotenv()


class GetCatImageTool(Tool):
    name = "get_cat_image"
    description = "Get a cat image"
    inputs = {}
    output_type = "image"

    def __init__(self):
        super().__init__()
        self.url = "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"

    def forward(self):
        from io import BytesIO

        import requests
        from PIL import Image

        response = requests.get(self.url)

        return Image.open(BytesIO(response.content))


get_cat_image = GetCatImageTool()

agent = CodeAgent(
    tools=[get_cat_image, VisitWebpageTool()],
    model=HfApiModel(),
    additional_authorized_imports=[
        "Pillow",
        "requests",
        "markdownify",
    ],  # "duckduckgo-search",
    use_e2b_executor=True,
)

agent.run(
    "Calculate how much is 2+2, then return me an image of a cat. Directly use the image provided in your state.",
    additional_args={"cat_image": get_cat_image()},
)  # Asking to directly return the image from state tests that additional_args are properly sent to server.

# Try the agent in a Gradio UI
from smolagents import GradioUI


GradioUI(agent).launch()
