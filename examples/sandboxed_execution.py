from smolagents import CodeAgent, InferenceClientModel, WebSearchTool


model = InferenceClientModel()

# Docker executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="docker") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

# E2B executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="e2b") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)

# WebAssembly executor example
with CodeAgent(tools=[], model=model, executor_type="wasm") as agent:
    output = agent.run("Calculate the square root of 125.")
print("Wasm executor result:", output)
# TODO: Support tools
# with CodeAgent(tools=[VisitWebpageTool()], model=model, executor_type="wasm") as agent:
#     output = agent.run("What is the content of the Wikipedia page at https://en.wikipedia.org/wiki/Intelligent_agent?")
