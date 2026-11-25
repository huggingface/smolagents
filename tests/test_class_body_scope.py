"""Class-body assignments should see earlier class-local bindings."""

from smolagents.local_python_executor import BASE_PYTHON_TOOLS, evaluate_python_code


def test_class_body_uses_local_binding():
    program = '''
value = "global"
class Settings:
    value = "local"
    alias = value
Settings.alias
'''
    result, _ = evaluate_python_code(
        program, static_tools=dict(BASE_PYTHON_TOOLS), state={}
    )
    assert result == "local"


def test_class_body_can_still_see_globals():
    program = '''
value = "global"
class Settings:
    alias = value
Settings.alias
'''
    result, _ = evaluate_python_code(
        program, static_tools=dict(BASE_PYTHON_TOOLS), state={}
    )
    assert result == "global"
