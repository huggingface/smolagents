# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from smolagents.gradio_ui import GradioUI
from smolagents.utils import encode_image_base64


def test_issue_2088(tmp_path, monkeypatch):
    monkeypatch.setattr("smolagents.gradio_ui._is_package_available", lambda package: package == "gradio")
    monkeypatch.setitem(sys.modules, "gradio", SimpleNamespace(ChatMessage=Mock()))

    uploaded_image_path = tmp_path / "uploaded.png"
    Image.new("RGB", (10, 10), color="red").save(uploaded_image_path)

    mock_agent = Mock()
    mock_agent.name = "test-agent"
    mock_agent.description = None
    mock_agent.stream_outputs = True

    def fake_run(task, images, stream, reset, additional_args):
        assert "Describe this image" in task
        assert stream is True
        assert reset is False
        assert additional_args is None
        assert len(images) == 1
        encode_image_base64(images[0])
        return []

    mock_agent.run.side_effect = fake_run

    ui = GradioUI(agent=mock_agent, file_upload_folder=str(tmp_path / "uploads"))

    responses = list(
        ui._stream_response(
            {"text": "Describe this image", "files": [str(uploaded_image_path)]},
            history=[],
        )
    )

    assert responses == []
