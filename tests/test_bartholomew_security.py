# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team & Bartholomew Trust Protocol. All rights reserved.
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
import unittest
import sys
from pathlib import Path

# Add src and src/smolagents to path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "smolagents"))

try:
    from smolagents.security import BartholomewASTSecurityGate, get_default_security_gate
except ImportError:
    import security
    BartholomewASTSecurityGate = security.BartholomewASTSecurityGate
    get_default_security_gate = security.get_default_security_gate



class BartholomewSecurityTestCase(unittest.TestCase):
    def setUp(self):
        self.gate = BartholomewASTSecurityGate(fail_closed=True)

    def test_safe_python_code_passes(self):
        code = "def compute_sum(a, b):\n    return a + b\nresult = compute_sum(10, 20)"
        is_safe, reason, receipt = self.gate.validate_code(code)
        self.assertTrue(is_safe)
        self.assertEqual(receipt["verdict"], "ALLOW")
        self.assertTrue("receipt_hash" in receipt)
        self.assertLess(receipt["latency_us"], 5000)  # Sub-millisecond execution

    def test_destructive_rm_rf_veto(self):
        code = "import os\nos.system('rm -rf /')"
        with self.assertRaises(PermissionError) as ctx:
            self.gate.validate_code(code)
        self.assertIn("BTP-SMOLAGENTS-VETO", str(ctx.exception))
        self.assertIn("BTP-AST-001", str(ctx.exception))

    def test_fork_bomb_veto(self):
        code = "import subprocess\nsubprocess.run(':(){ :|:& };:', shell=True)"
        with self.assertRaises(PermissionError) as ctx:
            self.gate.validate_code(code)
        self.assertIn("BTP-AST-001", str(ctx.exception))

    def test_credential_leak_veto(self):
        code = "api_key = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'"
        with self.assertRaises(PermissionError) as ctx:
            self.gate.validate_code(code)
        self.assertIn("BTP-CRED-001", str(ctx.exception))

    def test_tool_wrapper_protection(self):
        @self.gate.wrap_tool
        def custom_tool(query: str):
            return f"Queried: {query}"

        # Safe invocation
        self.assertEqual(custom_tool("machine learning papers"), "Queried: machine learning papers")

        # Dangerous invocation
        with self.assertRaises(PermissionError):
            custom_tool("rm -rf /etc/hosts")


if __name__ == "__main__":
    unittest.main()
