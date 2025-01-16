from smolagents import LiteLLMModel, ToolCallingAgent
from smolagents.extra_tools.ip import GetPublicIPTool
from smolagents.extra_tools.location import GetIPLocationTool
from smolagents.extra_tools.earthquake import GetEarthquakesTool

def setup_earthquake_agent(model_id: str="anthropic/claude-3-sonnet-20240229", verbose=False):
    model = LiteLLMModel(model_id=model_id)
    
    ip_tool = GetPublicIPTool()
    location_tool = GetIPLocationTool()
    earthquake_tool = GetEarthquakesTool()
    
    return ToolCallingAgent(
        tools=[ip_tool, location_tool, earthquake_tool],
        model=model,
        verbose=verbose,
        add_base_tools=False
    )

def main():
    agent = setup_earthquake_agent(verbose=True)
    query = """
        What's the nearest earthquake to my current location? 
        Please:
        1. Get my IP address
        2. Use it to determine my location
        3. Find the nearest recent earthquake
        4. Tell me its magnitude, location, approximate distance from me, and how long ago it occurred
    """
    print(f"\nQuerying: {query}\n")
    result = agent.run(query)
    print(f"\nResult: {result}\n")

if __name__ == "__main__":
    main()
