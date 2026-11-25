"""AgentImage/AgentAudio state refs must validate as image/audio inputs."""

from PIL import Image

from smolagents import AgentImage, Model, Tool, ToolCallingAgent
from smolagents._function_type_hints_utils import _get_json_schema_type


class ImageSize(Tool):
    name = "image_size"
    description = "Return an image's dimensions."
    inputs = {"image": {"type": "image", "description": "Input image."}}
    output_type = "array"

    def forward(self, image):
        return list(image.size)


class StubModel(Model):
    def generate(self, messages, **kwargs):
        raise AssertionError("not used")


def test_agent_image_maps_to_image_type():
    assert _get_json_schema_type(AgentImage) == {"type": "image"}


def test_tool_calling_agent_accepts_image_state_ref():
    image = AgentImage(Image.new("RGB", (3, 2)))
    agent = ToolCallingAgent(tools=[ImageSize()], model=StubModel(), verbosity_level=-1)
    agent.state["image.png"] = image

    result = agent.execute_tool_call("image_size", {"image": "image.png"})

    assert result == [3, 2]
