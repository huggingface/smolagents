import yaml
import os
from smolagents import GradioUI, CodeAgent, HfApiModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from managed_agents.useless.tools.final_answer import FinalAnswerTool as FinalAnswer



model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
provider=None,
)

final_answer = FinalAnswer()


with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent_useless = CodeAgent(
    model=model,
    tools=[],
    managed_agents=[],
    max_steps=20,
    verbosity_level=-1,
    grammar=None,
    planning_interval=None,
    name='useless',
    description='does nothing in particular',
    executor_type='local',
    executor_kwargs={},
    max_print_outputs_length=None,
    prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI(agent_useless).launch()
