from smolagents import CodeAgent, InferenceClientModel, WebSearchTool


model = InferenceClientModel()

# Blaxel executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="blaxel") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Blaxel executor result:", output)

# Docker executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="docker") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

# E2B executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="e2b") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)

# Modal executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="modal") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Modal executor result:", output)
