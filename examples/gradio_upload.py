from smolagents import (
    CodeAgent,
    HfApiModel,
    GradioUI
)

agent = CodeAgent(
    tools=[], model=HfApiModel(), max_steps=4, verbosity_level=0
)

GradioUI(agent, file_upload_folder='./data').launch()
