from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel


model = HfApiModel()

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, executor_type="docker")
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, executor_type="e2b")
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)
