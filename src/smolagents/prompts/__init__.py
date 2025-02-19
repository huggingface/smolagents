import os

import yaml


BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "code_agent.yaml"), "r") as f:
    CODE_SYSTEM_PROMPT = yaml.safe_load(f)

with open(os.path.join(BASE_DIR, "toolcalling_agent.yaml"), "r") as f:
    TOOLCALLING_SYSTEM_PROMPT = yaml.safe_load(f)

