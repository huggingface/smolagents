from unittest.mock import Mock, patch

import pytest

import smolagents
from smolagents import ChatMessage, CodeAgent, InferenceClientModel, LocalPythonExecutor, Model
from smolagents.local_python_executor import CodeOutput, PythonExecutor
from smolagents.remote_executors import RemotePythonExecutor
from smolagents.serialization import SerializationError


class PluginExecutor(LocalPythonExecutor):
    def __init__(self, additional_authorized_imports, logger, **kwargs):
        super().__init__(additional_authorized_imports, **kwargs)
        self.logger = logger
        self.cleaned_up = False

    def cleanup(self):
        self.cleaned_up = True


def create_executor(additional_authorized_imports, logger, **kwargs):
    return PluginExecutor(additional_authorized_imports, logger, **kwargs)


@pytest.fixture
def plugin_distribution(tmp_path, monkeypatch):
    dist_info = tmp_path / "test_executor-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: test-executor\nVersion: 1.0\n")
    (dist_info / "entry_points.txt").write_text(
        "[smolagents.executors]\n"
        "test-executor = tests.test_executor_plugins:PluginExecutor\n"
        "factory-executor = tests.test_executor_plugins:create_executor\n"
        "broken-executor = missing_executor_package:Executor\n"
        "local = missing_executor_package:Executor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))


@pytest.mark.parametrize("executor_type", ["test-executor", "factory-executor"])
def test_plugin_runs_with_tools_variables_and_cleanup(plugin_distribution, executor_type):
    model = Mock(spec=Model)
    model.generate.return_value = ChatMessage(
        role="assistant", content="<code>print(value)\nfinal_answer(value + 1)</code>"
    )
    with CodeAgent(
        tools=[],
        model=model,
        executor_type=executor_type,
        additional_authorized_imports=["decimal"],
        executor_kwargs={"max_print_outputs_length": 123},
    ) as agent:
        assert agent.run("Add one", additional_args={"value": 41}) == 42
        assert agent.memory.steps[-1].observations == "Execution logs:\n41\nLast output from code snippet:\n42"
        assert agent.python_executor.logger is agent.logger
        assert "decimal" in agent.python_executor.authorized_imports
        assert agent.python_executor.max_print_outputs_length == 123
    assert agent.python_executor.cleaned_up


def test_plugin_with_managed_agent(plugin_distribution):
    model = Mock(spec=Model)
    model.generate.return_value = ChatMessage(role="assistant", content='<code>final_answer(helper("answer"))</code>')
    helper = CodeAgent(tools=[], model=Mock(spec=Model), name="helper", description="Answer questions")
    helper.run = Mock(return_value="forty-two")
    agent = CodeAgent(tools=[], model=model, executor_type="test-executor", managed_agents=[helper])
    assert "forty-two" in agent.run("Ask helper")
    helper.run.assert_called_once()


@pytest.mark.parametrize("executor_type", ["local", "blaxel", "e2b", "modal", "docker"])
def test_builtins_bypass_plugin_discovery(executor_type, plugin_distribution):
    with patch("importlib.metadata.entry_points", side_effect=AssertionError("Unexpected discovery")):
        if executor_type == "local":
            agent = CodeAgent(tools=[], model=Model(), executor_kwargs={"max_print_outputs_length": 123})
            assert isinstance(agent.python_executor, LocalPythonExecutor)
            assert agent.python_executor.max_print_outputs_length == 123
        else:
            executor_class = {
                "blaxel": "BlaxelExecutor",
                "e2b": "E2BExecutor",
                "modal": "ModalExecutor",
                "docker": "DockerExecutor",
            }[executor_type]
            with patch(f"smolagents.agents.{executor_class}") as factory:
                agent = CodeAgent(
                    tools=[], model=Model(), executor_type=executor_type, executor_kwargs={"allow_pickle": True}
                )
                factory.assert_called_once_with([], agent.logger, allow_pickle=True)
                helper = CodeAgent(tools=[], model=Model(), name="helper", description="Answer questions")
                with pytest.raises(Exception, match="Managed agents are not yet supported"):
                    CodeAgent(tools=[], model=Model(), executor_type=executor_type, managed_agents=[helper])
                assert factory.call_count == 1


def test_direct_executor_takes_precedence():
    class FalseyExecutor(LocalPythonExecutor):
        def __bool__(self):
            return False

    executor = FalseyExecutor([])
    model = Mock(spec=Model)
    model.generate.return_value = ChatMessage(role="assistant", content="<code>final_answer(value + 1)</code>")
    with patch("importlib.metadata.entry_points", side_effect=AssertionError("Unexpected discovery")):
        agent = CodeAgent(tools=[], model=model, executor=executor, executor_type="missing", max_steps=1)
        assert agent.run("Add one", additional_args={"value": 41}) == 42
    assert agent.python_executor is executor


def test_plugin_serialization(plugin_distribution):
    agent = CodeAgent(
        tools=[],
        model=InferenceClientModel(),
        executor_type="test-executor",
        executor_kwargs={"max_print_outputs_length": 123},
    )
    saved = agent.to_dict()
    restored = CodeAgent.from_dict(saved)
    assert restored.executor_type == "test-executor"
    assert restored.executor_kwargs == {"max_print_outputs_length": 123}
    assert isinstance(restored.python_executor, PluginExecutor)
    assert restored.python_executor.max_print_outputs_length == 123
    assert isinstance(CodeAgent.from_dict(saved, executor_type="local").python_executor, LocalPythonExecutor)


def test_unknown_plugin_does_not_import_module(plugin_distribution):
    with pytest.raises(ValueError, match="Unsupported executor type: os:system"):
        CodeAgent(tools=[], model=Model(), executor_type="os:system")


def test_duplicate_plugin_names_fail_before_load():
    entry_points = [Mock(), Mock()]
    with patch("importlib.metadata.entry_points", return_value=entry_points):
        with pytest.raises(ValueError, match="Multiple.*test-executor"):
            CodeAgent(tools=[], model=Model(), executor_type="test-executor")
    for entry_point in entry_points:
        entry_point.load.assert_not_called()


def test_duplicate_installed_plugins(plugin_distribution, tmp_path):
    dist_info = tmp_path / "other_executor-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: other-executor\nVersion: 1.0\n")
    (dist_info / "entry_points.txt").write_text(
        "[smolagents.executors]\ntest-executor = missing_executor_package:Executor\n"
    )
    with pytest.raises(ValueError, match="Multiple.*test-executor"):
        CodeAgent(tools=[], model=Model(), executor_type="test-executor")


@pytest.mark.parametrize("factory, message", [(42, "callable"), (lambda *args, **kwargs: object(), "PythonExecutor")])
def test_invalid_plugin(factory, message):
    entry_point = Mock()
    entry_point.load.return_value = factory
    with patch("importlib.metadata.entry_points", return_value=[entry_point]):
        with pytest.raises(TypeError, match=message):
            CodeAgent(tools=[], model=Model(), executor_type="test-executor")


def test_plugin_errors_propagate(plugin_distribution):
    with pytest.raises(ModuleNotFoundError, match="missing_executor_package"):
        CodeAgent(tools=[], model=Model(), executor_type="broken-executor")
    failure = RuntimeError("provider could not start")
    entry_point = Mock()
    entry_point.load.return_value = Mock(side_effect=failure)
    with patch("importlib.metadata.entry_points", return_value=[entry_point]):
        with pytest.raises(RuntimeError) as error:
            CodeAgent(tools=[], model=Model(), executor_type="test-executor")
    assert error.value is failure


def test_public_executor_contract():
    assert smolagents.PythonExecutor is PythonExecutor
    assert smolagents.CodeOutput is CodeOutput
    assert smolagents.RemotePythonExecutor is RemotePythonExecutor
    assert RemotePythonExecutor.deserialize_final_answer('safe:{"answer": 42}') == {"answer": 42}
    with pytest.raises(SerializationError, match="Pickle data rejected"):
        RemotePythonExecutor.deserialize_final_answer("pickle:invalid")
    with pytest.raises(SerializationError, match="Unknown final answer format"):
        RemotePythonExecutor.deserialize_final_answer("unprefixed")
