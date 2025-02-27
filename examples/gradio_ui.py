from smolagents import CodeAgent, GradioUI, HfApiModel


agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    max_steps=4,
    verbosity_level=1,
    name="example_agent",
    description="This is an example agent that has no tools and uses only code."
)

GradioUI(agent, file_upload_folder="./data").launch()
