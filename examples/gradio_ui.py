from smolagents import CodeAgent, GradioUI, InferenceClientModel


agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=False,
)

GradioUI(agent, file_upload_folder="./data").launch()
