#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


SKILL_FILENAME = "SKILL.md"
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
SKILL_NAME_REGEX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class AgentSkill:
    """Structured representation of one local skill definition."""

    name: str
    description: str
    path: Path
    content: str


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Invalid skill file at '{path}': missing YAML frontmatter delimiter '---'.")

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = "\n".join(lines[1:index]).strip()
            body = "\n".join(lines[index + 1 :]).strip()
            return frontmatter, body

    raise ValueError(f"Invalid skill file at '{path}': missing closing YAML frontmatter delimiter '---'.")


def _validate_skill_name(name: str, path: Path) -> None:
    if len(name) > MAX_SKILL_NAME_LENGTH:
        raise ValueError(
            f"Invalid skill name in '{path}': maximum length is {MAX_SKILL_NAME_LENGTH} characters, got {len(name)}."
        )
    if not SKILL_NAME_REGEX.match(name):
        raise ValueError(
            f"Invalid skill name in '{path}': expected lowercase alphanumeric segments separated by single hyphens."
        )


def _validate_skill_description(description: str, path: Path) -> None:
    if not description:
        raise ValueError(f"Invalid skill description in '{path}': field cannot be empty.")
    if len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
        raise ValueError(
            f"Invalid skill description in '{path}': maximum length is {MAX_SKILL_DESCRIPTION_LENGTH} characters."
        )


def _normalize_skill_path(skill_source: str | Path) -> Path:
    skill_path = Path(skill_source).expanduser()
    if skill_path.is_dir():
        skill_path = skill_path / SKILL_FILENAME
    if skill_path.name != SKILL_FILENAME:
        raise ValueError(
            f"Invalid skill path '{skill_source}': expected a '{SKILL_FILENAME}' file or a directory containing it."
        )
    if not skill_path.is_file():
        raise FileNotFoundError(f"Skill file not found at '{skill_path}'.")
    return skill_path.resolve()


def parse_agent_skill(skill_source: str | Path) -> AgentSkill:
    """Parse a local SKILL.md file into an AgentSkill."""

    skill_path = _normalize_skill_path(skill_source)
    text = skill_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, skill_path)

    metadata = yaml.safe_load(frontmatter)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid skill metadata in '{skill_path}': expected a YAML object.")

    name = str(metadata.get("name", "")).strip()
    description = str(metadata.get("description", "")).strip()

    if not name:
        raise ValueError(f"Invalid skill metadata in '{skill_path}': missing required field 'name'.")
    if not description:
        raise ValueError(f"Invalid skill metadata in '{skill_path}': missing required field 'description'.")

    _validate_skill_name(name, skill_path)
    _validate_skill_description(description, skill_path)

    return AgentSkill(name=name, description=description, path=skill_path, content=body)


def load_agent_skills(skill_sources: list[str | Path] | None = None) -> list[AgentSkill]:
    """Load and validate a collection of local skills."""

    if not skill_sources:
        return []

    skills: list[AgentSkill] = []
    seen_names: set[str] = set()
    for source in skill_sources:
        skill = parse_agent_skill(source)
        normalized_name = skill.name.lower()
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate skill name detected: '{skill.name}'.")
        seen_names.add(normalized_name)
        skills.append(skill)
    return skills


def select_skills_for_task(skills: list[AgentSkill], task: str | None) -> list[AgentSkill]:
    """
    Select skills explicitly mentioned in a task.

    A skill is selected if the task includes either `$skill-name` or `skill-name`.
    """

    if not skills or not task:
        return []

    task_lower = task.lower()
    selected_skills: list[AgentSkill] = []

    for skill in skills:
        skill_name = skill.name.lower()
        has_sigil_mention = f"${skill_name}" in task_lower
        has_plain_mention = (
            re.search(rf"(?<![a-z0-9-]){re.escape(skill_name)}(?![a-z0-9-])", task_lower) is not None
        )
        if has_sigil_mention or has_plain_mention:
            selected_skills.append(skill)

    return selected_skills
