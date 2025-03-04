from smolagents import CodeAgent, GradioUI, HfApiModel
from smolagents.monitoring import LogLevel


agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    max_steps=4,
    verbosity_level=LogLevel.INFO,
    name="example_agent",
    description="This is an example agent that has no tools and uses only code.",
)

GradioUI(agent, file_upload_folder="./data").launch()
