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
import os
import tempfile
import unittest
import uuid

import PIL.Image

from smolagents.agent_types import AgentAudio, AgentImage, AgentText

from .utils.markers import require_soundfile, require_torch


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@require_soundfile
@require_torch
class AgentAudioTests(unittest.TestCase):
    def test_from_tensor(self):
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        agent_type = AgentAudio(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))

        del agent_type

        # Ensure the path remains even after the object deletion
        self.assertTrue(os.path.exists(path))

        # Ensure that the file contains the same value as the original tensor
        new_tensor, _ = sf.read(path)
        self.assertTrue(torch.allclose(tensor, torch.tensor(new_tensor), atol=1e-4))

    def test_from_string(self):
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        path = get_new_path(suffix=".wav")
        sf.write(path, tensor, 16000)

        agent_type = AgentAudio(path)

        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))
        self.assertEqual(agent_type.to_string(), path)


class TestAgentImage:
    def test_from_numpy(self):
        import numpy as np

        # Test uint8 ndarray
        array = np.zeros((32, 32, 3), dtype=np.uint8)
        array[0, 0] = [255, 128, 64]
        agent_type = AgentImage(array)
        raw_img = agent_type.to_raw()
        assert isinstance(raw_img, PIL.Image.Image)
        res_arr = np.array(raw_img)
        assert np.array_equal(res_arr[0, 0], [255, 128, 64])

        # Test float ndarray in [0.0, 1.0]
        f_array = np.ones((16, 16, 3), dtype=np.float32)
        agent_type_f = AgentImage(f_array)
        raw_img_f = agent_type_f.to_raw()
        assert np.array_equal(np.array(raw_img_f)[0, 0], [255, 255, 255])

    def test_from_tensor(self):
        import numpy as np
        import torch

        tensor = torch.zeros((64, 64, 3), dtype=torch.uint8)
        tensor[0, 0] = torch.tensor([255, 128, 0], dtype=torch.uint8)
        agent_type = AgentImage(tensor)
        path = str(agent_type.to_string())

        assert isinstance(agent_type.to_raw(), PIL.Image.Image)
        res_arr = np.array(agent_type.to_raw())
        assert np.array_equal(res_arr[0, 0], [255, 128, 0])

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_string(self, tmp_path):
        image_path = str(tmp_path / "test_img.png")
        image = PIL.Image.new("RGB", (32, 32), color="red")
        image.save(image_path)

        agent_type = AgentImage(image_path)
        assert agent_type.to_string() == image_path
        assert isinstance(agent_type.to_raw(), PIL.Image.Image)

    def test_from_image(self):
        image = PIL.Image.new("RGB", (32, 32), color="blue")
        agent_type = AgentImage(image)

        assert isinstance(agent_type.to_raw(), PIL.Image.Image)
        path = agent_type.to_string()
        assert os.path.exists(path)


class AgentTextTests(unittest.TestCase):
    def test_from_string(self):
        string = "Hey!"
        agent_type = AgentText(string)

        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
