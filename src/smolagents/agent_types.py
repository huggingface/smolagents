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
import logging
import os
import pathlib
import tempfile
import uuid
from io import BytesIO

import PIL.Image
import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    """
    Abstract class to be reimplemented to define types that can be returned by agents.

    These objects serve three purposes:

    - They behave as they were the type they're meant to be, e.g., a string for text, a PIL.Image.Image for images
    - They can be stringified: str(object) in order to return a string defining the object
    - They should be displayed correctly in ipython notebooks/colab/jupyter
    """

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return self.to_string()

    def to_raw(self):
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    """
    Text type returned by the agent. Behaves as a string.
    """

    def to_raw(self):
        return self._value

    def to_string(self):
        return str(self._value)


class AgentImage(AgentType, PIL.Image.Image):
    """
    Image type returned by the agent. Behaves as a PIL.Image.Image.
    """

    def __init__(self, value):
        AgentType.__init__(self, value)
        PIL.Image.Image.__init__(self)

        self._path = None
        self._raw = None
        self._tensor = None

        if isinstance(value, AgentImage):
            self._raw, self._path, self._tensor = value._raw, value._path, value._tensor
        elif isinstance(value, PIL.Image.Image):
            self._raw = value
        elif isinstance(value, bytes):
            self._raw = PIL.Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            self._path = value
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    self._tensor = value
                import numpy as np

                if isinstance(value, np.ndarray):
                    self._tensor = torch.from_numpy(value)
            except ModuleNotFoundError:
                pass

        if self._path is None and self._raw is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Image, display

        display(Image(self.to_string()))

    def to_raw(self):
        """
        Returns the "raw" version of that object. In the case of an AgentImage, it is a PIL.Image.Image.
        """
        if self._raw is not None:
            return self._raw

        if self._path is not None:
            self._raw = PIL.Image.open(self._path)
            return self._raw

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()
            return PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
        version of the image.
        """
        if self._path is not None:
            return self._path

        if self._raw is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            self._raw.save(self._path, format="png")
            return self._path

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()

            # There is likely simpler than load into image into save
            img = PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            img.save(self._path, format="png")

            return self._path

    def save(self, output_bytes, format: str = None, **params):
        """
        Saves the image to a file.
        Args:
            output_bytes (bytes): The output bytes to save the image to.
            format (str): The format to use for the output image. The format is the same as in PIL.Image.save.
            **params: Additional parameters to pass to PIL.Image.save.
        """
        img = self.to_raw()
        img.save(output_bytes, format=format, **params)


class AgentAudio(AgentType, str):
    """
    Audio type returned by the agent.
    """

    def __init__(self, value, samplerate=16_000):
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'smolagents[audio]'`"
            )
        import numpy as np
        import torch

        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        elif isinstance(value, tuple):
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                self._tensor = torch.from_numpy(value[1])
            else:
                self._tensor = torch.tensor(value[1])
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
        import soundfile as sf

        if self._tensor is not None:
            return self._tensor

        import torch

        if self._path is not None:
            if "://" in str(self._path):
                response = requests.get(self._path)
                response.raise_for_status()
                tensor, self.samplerate = sf.read(BytesIO(response.content))
            else:
                tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        import soundfile as sf

        if self._path is not None:
            return self._path

        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path


_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args, **kwargs):
    args = [(arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args]
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    return args, kwargs


def handle_agent_output_types(output, output_type=None):
    if output_type in _AGENT_TYPE_MAPPING:
        # If the class has defined outputs, we can map directly according to the class definition
        decoded_outputs = _AGENT_TYPE_MAPPING[output_type](output)
        return decoded_outputs

    # If the class does not have defined output, then we map according to the type
    if isinstance(output, str):
        return AgentText(output)
    if isinstance(output, PIL.Image.Image):
        return AgentImage(output)
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return AgentAudio(output)
    except ModuleNotFoundError:
        pass
    return output


__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]
