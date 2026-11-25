"""Tool source export must handle property/classmethod/staticmethod descriptors."""

import ast

from smolagents import Tool
from smolagents.utils import instance_to_source


class GreetingTool(Tool):
    name = "greeting"
    description = "Return a greeting."
    inputs = {}
    output_type = "string"

    @property
    def suffix(self):
        return "!"

    @classmethod
    def helper(cls, value):
        return value.upper()

    @staticmethod
    def neutral():
        return "neutral"

    def forward(self):
        return "hello" + self.suffix


def test_descriptors_exported_as_valid_python():
    source = instance_to_source(GreetingTool(), base_cls=Tool)

    ast.parse(source)
    assert "@property" in source
    assert "@classmethod" in source
    assert "@staticmethod" in source
    assert "<property object" not in source
    assert "<classmethod" not in source

    namespace = {}
    exec(source, namespace)
    recreated = namespace["GreetingTool"]
    instance = recreated()
    assert instance.suffix == "!"
    assert recreated.helper("ab") == "AB"
    assert recreated.neutral() == "neutral"
