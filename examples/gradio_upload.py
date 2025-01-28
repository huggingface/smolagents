from smolagents import CodeAgent, GradioUI, HfApiModel


agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=4, verbosity_level=1)

GradioUI(agent, file_upload_folder="./data").launch()
