from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel


model = InferenceClientModel()

# Docker executor example
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, executor_type="docker")
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

# E2B executor example
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, executor_type="e2b")
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)

# WebAssembly executor example
agent = CodeAgent(tools=[], model=model, executor_type="wasm")
output = agent.run("Calculate the square root of 125.")
print("Wasm executor result:", output)
# TODO: Support tools
# agent = CodeAgent(tools=[VisitWebpageTool()], model=model, executor_type="wasm")
# output = agent.run("What is the content of the Wikipedia page at https://en.wikipedia.org/wiki/Intelligent_agent?")
