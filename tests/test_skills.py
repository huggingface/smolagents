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
from textwrap import dedent

import pytest

from smolagents.skills import load_agent_skills, parse_agent_skill, select_skills_for_task


def _write_skill(tmp_path, name: str, description: str, body: str):
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        dedent(
            f"""
            ---
            name: {name}
            description: {description}
            ---

            {body}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return skill_dir


def test_parse_agent_skill_from_directory(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        name="release-manager",
        description="Create release notes and version plans.",
        body="Use this skill for release preparation tasks.",
    )
    skill = parse_agent_skill(skill_dir)

    assert skill.name == "release-manager"
    assert skill.description == "Create release notes and version plans."
    assert skill.path == (skill_dir / "SKILL.md").resolve()
    assert "release preparation" in skill.content


def test_load_agent_skills_rejects_duplicate_names(tmp_path):
    first_skill = _write_skill(
        tmp_path / "one",
        name="qa-review",
        description="Review pull requests for regressions.",
        body="Review behavior changes first.",
    )
    second_skill = _write_skill(
        tmp_path / "two",
        name="qa-review",
        description="Second definition with duplicated name.",
        body="This should fail.",
    )

    with pytest.raises(ValueError, match="Duplicate skill name"):
        load_agent_skills([first_skill, second_skill])


def test_parse_agent_skill_requires_frontmatter(tmp_path):
    skill_dir = tmp_path / "invalid-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("No frontmatter", encoding="utf-8")

    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        parse_agent_skill(skill_dir)


def test_select_skills_for_task_supports_sigil_and_plain_mentions(tmp_path):
    first_skill = _write_skill(
        tmp_path,
        name="security-audit",
        description="Audit code for security issues.",
        body="Check for auth and input validation gaps.",
    )
    second_skill = _write_skill(
        tmp_path,
        name="perf-review",
        description="Review runtime performance.",
        body="Look for hot loops and expensive allocations.",
    )
    loaded_skills = load_agent_skills([first_skill, second_skill])

    selected = select_skills_for_task(
        loaded_skills,
        "Please run $security-audit and then do a perf-review before merging.",
    )

    assert [skill.name for skill in selected] == ["security-audit", "perf-review"]
