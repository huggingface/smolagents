import sys
import types

from smolagents import Tool


def test_issue_2481(monkeypatch):
    created_clients = []

    class GradioClient25Client:
        def __init__(self, space_id, token=None):
            self.space_id = space_id
            self.token = token
            created_clients.append(self)

        def view_api(self, return_format=None, print_info=None):
            return {
                "named_endpoints": {
                    "/predict": {
                        "parameters": [
                            {
                                "parameter_name": "prompt",
                                "type": {"type": "string"},
                                "python_type": {"description": "Image prompt"},
                                "parameter_has_default": False,
                            }
                        ],
                        "returns": [{"component": "Image"}],
                    }
                }
            }

    gradio_client_module = types.ModuleType("gradio_client")
    gradio_client_module.Client = GradioClient25Client
    gradio_client_module.handle_file = lambda file: file
    monkeypatch.setitem(sys.modules, "gradio_client", gradio_client_module)

    tool = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="Generate an image from a prompt",
        token="test_hf_token",
    )

    assert tool.name == "image_generator"
    assert tool.description == "Generate an image from a prompt"
    assert tool.inputs == {
        "prompt": {"type": "string", "description": "Image prompt", "nullable": False}
    }
    assert tool.output_type == "image"
    assert len(created_clients) == 1
    assert created_clients[0].space_id == "black-forest-labs/FLUX.1-schnell"
    assert created_clients[0].token == "test_hf_token"
