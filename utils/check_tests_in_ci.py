# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. team.
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
"""Check that all tests are called in CI."""

from pathlib import Path


ROOT = Path(__file__).parent.parent

TESTS_FOLDER = ROOT / "tests"
CI_WORKFLOW_FILE = ROOT / ".github" / "workflows" / "tests.yml"


def check_tests_in_ci():
    """List all test files in `./tests/` and check if they are listed in the CI workflow.

    Since each test file is triggered separately in the CI workflow, it is easy to forget a new one when adding new
    tests, hence this check.

    NOTE: current implementation is quite naive but should work for now. Must be updated if one want to ignore some
          tests or if file naming is updated (currently only files starting by `test_*` are checked)
    """
    test_files = [
        path.relative_to(TESTS_FOLDER).as_posix()
        for path in TESTS_FOLDER.glob("**/*.py")
        if path.name.startswith("test_")
    ]
    ci_workflow_file_content = CI_WORKFLOW_FILE.read_text()
    missing_test_files = [test_file for test_file in test_files if test_file not in ci_workflow_file_content]
    if missing_test_files:
        print(
            "❌ Some test files seem to be ignored in the CI:\n"
            + "\n".join(f"   - {test_file}" for test_file in missing_test_files)
            + f"\n   Please add them manually in {CI_WORKFLOW_FILE}."
        )
        exit(1)
    else:
        print("✅ All good!")
        exit(0)


if __name__ == "__main__":
    check_tests_in_ci()
