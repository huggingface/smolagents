import yaml
import os
from smolagents import GradioUI, ToolCallingAgent, HfApiModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from managed_agents.web_agent.tools.web_search import DuckDuckGoSearchTool as WebSearch
from managed_agents.web_agent.tools.visit_webpage import VisitWebpageTool as VisitWebpage
from managed_agents.web_agent.tools.final_answer import FinalAnswerTool as FinalAnswer



model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
provider=None,
)

web_search = WebSearch()
visit_webpage = VisitWebpage()
final_answer = FinalAnswer()


with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent_web_agent = ToolCallingAgent(
    model=model,
    tools=[web_search, visit_webpage],
    managed_agents=[],
    max_steps=20,
    verbosity_level=-1,
    grammar=None,
    planning_interval=None,
    name='web_agent',
    description='does web searches',
    prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI(agent_web_agent).launch()
