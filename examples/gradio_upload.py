from smolagents import (
    CodeAgent,
    HfApiModel,
    GradioUI
)

agent = CodeAgent(
    tools=[], model=HfApiModel(), max_steps=4, verbose=True
)

GradioUI(agent, UPLOAD_FOLDER='./data').launch()
