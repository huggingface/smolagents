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

import numpy as np
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


@require_torch
class TestAgentImage:
    def test_from_tensor(self):
        import torch

        tensor = torch.tensor(
            [
                [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]],
                [[0.25, 0.75, 0.125], [0.875, 0.375, 0.625]],
            ]
        )
        agent_type = AgentImage(tensor)
        expected = (tensor.numpy() * 255).astype(np.uint8)

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        path = str(agent_type.to_string())

        assert np.array_equal(np.asarray(PIL.Image.open(path)), expected)

        # Ensure that the tensor and the agent_type's tensor are the same
        assert torch.allclose(tensor, agent_type._tensor, atol=1e-4)

        assert isinstance(agent_type.to_raw(), PIL.Image.Image)

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_float_tensor_with_0_to_255_values(self):
        import torch

        tensor = torch.tensor(
            [[[0.0, 128.0, 255.0], [255.0, 128.0, 0.0]]],
            dtype=torch.float32,
        )
        agent_type = AgentImage(tensor)
        expected = tensor.numpy().astype(np.uint8)

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        assert np.array_equal(np.asarray(PIL.Image.open(agent_type.to_string())), expected)

    def test_from_float_tensor_clips_out_of_range_values(self):
        import torch

        tensor = torch.tensor([[[-1.0, 128.0, 256.0]]], dtype=torch.float32)
        agent_type = AgentImage(tensor)
        expected = np.array([[[0, 128, 255]]], dtype=np.uint8)

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        assert np.array_equal(np.asarray(PIL.Image.open(agent_type.to_string())), expected)

    def test_from_float_tensor_with_value_above_one_uses_0_to_255_range(self):
        import torch

        tensor = torch.tensor([[[0.0, 0.5, 1.5]]], dtype=torch.float32)
        agent_type = AgentImage(tensor)
        expected = np.array([[[0, 0, 1]]], dtype=np.uint8)

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        assert np.array_equal(np.asarray(PIL.Image.open(agent_type.to_string())), expected)

    def test_from_float_tensor_allows_small_overshoot_above_one(self):
        import torch

        tensor = torch.tensor([[[0.0, 0.5, 1.0001]]], dtype=torch.float32)
        agent_type = AgentImage(tensor)
        expected = np.array([[[0, 127, 255]]], dtype=np.uint8)

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        assert np.array_equal(np.asarray(PIL.Image.open(agent_type.to_string())), expected)

    def test_from_uint8_tensor_preserves_values(self):
        import torch

        tensor = torch.tensor(
            [[[0, 128, 255], [255, 128, 0]]],
            dtype=torch.uint8,
        )
        agent_type = AgentImage(tensor)
        expected = tensor.numpy()

        assert np.array_equal(np.asarray(agent_type.to_raw()), expected)
        assert np.array_equal(np.asarray(PIL.Image.open(agent_type.to_string())), expected)

    def test_from_string(self, shared_datadir):
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(path)

        assert path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_image(self, shared_datadir):
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(image)

        assert not path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)


class AgentTextTests(unittest.TestCase):
    def test_from_string(self):
        string = "Hey!"
        agent_type = AgentText(string)

        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
