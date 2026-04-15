import json
import os
import tempfile
import unittest

from smolagents.agents import ALLOWED_CALLBACK_NAMESPACES, CALLBACK_REGISTRY, CodeAgent, MultiStepAgent
from smolagents.memory import ActionStep
from smolagents.models import LiteLLMModel


def my_test_callback(step):
    """A dummy callback for testing."""
    pass


def my_test_check(answer):
    """A dummy check for testing."""
    return True


class TestCallableSerialization(unittest.TestCase):
    def tearDown(self):
        # Clean up any registry entries
        CALLBACK_REGISTRY.pop("custom.my_test_callback", None)

    def test_roundtrip(self):
        """Test full save/load cycle verifying callbacks are preserved."""
        model = LiteLLMModel(model_id="gpt-4o", api_key="dummy")
        agent = CodeAgent(
            tools=[], model=model, step_callbacks=[my_test_callback], final_answer_checks=[my_test_check]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save
            agent.save(tmp_dir)

            # 1. Verify JSON content
            with open(os.path.join(tmp_dir, "agent.json"), "r") as f:
                data = json.load(f)

            # Check for serialized paths
            self.assertIn("final_answer_checks", data)
            self.assertIn("step_callbacks", data)

            # Depending on how the test is run, the module might be __main__ or the file name
            callback_path = f"{my_test_callback.__module__}.{my_test_callback.__qualname__}"
            check_path = f"{my_test_check.__module__}.{my_test_check.__qualname__}"

            self.assertIn(check_path, data["final_answer_checks"])
            self.assertIn("ActionStep", data["step_callbacks"])
            self.assertIn(callback_path, data["step_callbacks"]["ActionStep"])

            # 2. Verify Reload
            reloaded_agent = CodeAgent.from_folder(tmp_dir, allowed_callback_namespaces={"smolagents", "tests"})

            # Check checks
            loaded_checks = reloaded_agent.final_answer_checks
            self.assertEqual(len(loaded_checks), 1)
            self.assertEqual(loaded_checks[0].__name__, "my_test_check")

            # Check callbacks
            loaded_cbs = dict(reloaded_agent.step_callbacks.items())[ActionStep]
            # One should be our callback, one should be monitor.update_metrics
            cb_names = [cb.__name__ for cb in loaded_cbs]
            self.assertIn("my_test_callback", cb_names)
            self.assertIn("update_metrics", cb_names)

    def test_lambda_skipping(self):
        """Verify lambda functions are safely skipped during serialization."""
        model = LiteLLMModel(model_id="gpt-4o", api_key="dummy")
        agent = CodeAgent(tools=[], model=model, final_answer_checks=[lambda x: True])

        with tempfile.TemporaryDirectory() as tmp_dir:
            agent.save(tmp_dir)
            with open(os.path.join(tmp_dir, "agent.json"), "r") as f:
                data = json.load(f)

            # Lambda should be skipped
            self.assertEqual(len(data["final_answer_checks"]), 0)

    def test_security_whitelist_blocks_unsafe(self):
        """Verify unsafe module paths (e.g., os.system) are blocked."""
        unsafe_path = "os.system"
        res = MultiStepAgent._deserialize_callable(unsafe_path)
        self.assertIsNone(res)

    def test_security_whitelist_allows_smolagents(self):
        """Verify smolagents namespace is allowed."""
        safe_path = "smolagents.agents.MultiStepAgent"
        res = MultiStepAgent._deserialize_callable(safe_path)
        self.assertEqual(res, MultiStepAgent)

    def test_callback_registry_lookup(self):
        """Verify CALLBACK_REGISTRY is checked before namespace validation."""
        # Register a callback under a custom (normally blocked) path
        CALLBACK_REGISTRY["custom.my_test_callback"] = my_test_callback

        res = MultiStepAgent._deserialize_callable("custom.my_test_callback")
        self.assertEqual(res, my_test_callback)

    def test_allowed_namespaces_extension(self):
        """Verify that ALLOWED_CALLBACK_NAMESPACES can be extended for user packages."""
        ALLOWED_CALLBACK_NAMESPACES.add("tests")
        try:
            path = f"{my_test_callback.__module__}.{my_test_callback.__qualname__}"
            res = MultiStepAgent._deserialize_callable(path)
            self.assertEqual(res, my_test_callback)

            # Remove the namespace and verify it's now blocked
            ALLOWED_CALLBACK_NAMESPACES.discard("tests")
            res = MultiStepAgent._deserialize_callable(path)
            self.assertIsNone(res)
        finally:
            ALLOWED_CALLBACK_NAMESPACES.discard("tests")

    def test_instance_level_isolation(self):
        """Verify that setting allowed_callback_namespaces on one agent does not affect another."""
        model = LiteLLMModel(model_id="gpt-4o", api_key="dummy")
        # Agent 1 allows 'tests'
        agent1 = CodeAgent(
            tools=[],
            model=model,
            step_callbacks=[my_test_callback],
            allowed_callback_namespaces={"smolagents", "tests"},
        )
        # Agent 2 allows only 'smolagents'
        agent2 = CodeAgent(
            tools=[], model=model, step_callbacks=[my_test_callback], allowed_callback_namespaces={"smolagents"}
        )

        # Agent 1 should be able to serialize/deserialize it (if it used its own namespaces)
        # Note: _deserialize_callable is a classmethod, so we need to pass the namespaces manually
        # OR we check if the agent instance uses it in to_dict/from_dict logic

        with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
            agent1.save(tmp_dir1)
            agent2.save(tmp_dir2)

            # Loading agent1 with 'tests' whitelist should work
            loaded1 = CodeAgent.from_folder(tmp_dir1, allowed_callback_namespaces={"smolagents", "tests"})
            self.assertEqual(len(dict(loaded1.step_callbacks.items())[ActionStep]), 2)  # our cb + monitor

            # Loading agent1 WITHOUT 'tests' whitelist should fail to load our callback
            loaded1_restricted = CodeAgent.from_folder(tmp_dir1, allowed_callback_namespaces={"smolagents"})
            cb_names = [cb.__name__ for cb in dict(loaded1_restricted.step_callbacks.items()).get(ActionStep, [])]
            self.assertNotIn("my_test_callback", cb_names)

    def test_callback_registry_instance_level(self):
        """Verify instance-level callback_registry works."""
        model = LiteLLMModel(model_id="gpt-4o", api_key="dummy")
        custom_registry = {"custom.path": my_test_callback}
        agent = CodeAgent(tools=[], model=model, callback_registry=custom_registry, step_callbacks=[my_test_callback])

        with tempfile.TemporaryDirectory() as tmp_dir:
            agent.save(tmp_dir)

            # Should load if we pass the same registry
            loaded = CodeAgent.from_folder(tmp_dir, callback_registry=custom_registry)
            cb_names = [cb.__name__ for cb in dict(loaded.step_callbacks.items())[ActionStep]]
            self.assertIn("my_test_callback", cb_names)

            # Should NOT load if we don't pass the registry and it's not in global or whitelist
            loaded_no_reg = CodeAgent.from_folder(tmp_dir)
            cb_names_no_reg = [cb.__name__ for cb in dict(loaded_no_reg.step_callbacks.items()).get(ActionStep, [])]
            self.assertNotIn("my_test_callback", cb_names_no_reg)


if __name__ == "__main__":
    unittest.main()
