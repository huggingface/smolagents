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
    def setUp(self):
        # Allow the tests namespace for deserialization during tests
        ALLOWED_CALLBACK_NAMESPACES.add("tests")

    def tearDown(self):
        # Clean up: remove the tests namespace and any registry entries
        ALLOWED_CALLBACK_NAMESPACES.discard("tests")
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
            reloaded_agent = CodeAgent.from_folder(tmp_dir)

            # Check checks
            loaded_checks = reloaded_agent.final_answer_checks
            self.assertEqual(len(loaded_checks), 1)
            self.assertEqual(loaded_checks[0].__name__, "my_test_check")

            # Check callbacks
            loaded_cbs = reloaded_agent.step_callbacks._callbacks[ActionStep]
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
        # 'tests' was already added in setUp
        path = f"{my_test_callback.__module__}.{my_test_callback.__qualname__}"
        res = MultiStepAgent._deserialize_callable(path)
        self.assertEqual(res, my_test_callback)

        # Remove the namespace and verify it's now blocked
        ALLOWED_CALLBACK_NAMESPACES.discard("tests")
        res = MultiStepAgent._deserialize_callable(path)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
