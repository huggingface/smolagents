from smolagents import LiteLLMModel, ToolCallingAgent, MultiStepAgent
from smolagents import VisitWebpageTool, DuckDuckGoSearchTool


def setup_agent(model_id: str="anthropic/claude-3-5-sonnet-latest") -> MultiStepAgent:
    '''
    Returns a fully-configured Agent with the basic tools, using the supplied model_id
    (environment variables are assumed to be set up with the API keys you need for that model)
    '''
    model = LiteLLMModel(model_id=model_id)
    # code_agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    agent = ToolCallingAgent(tools=[], model=model, add_base_tools=True)
    return agent


def setup_simple_agent(model_id: str="anthropic/claude-3-5-sonnet-latest",
                       verbose=False) -> MultiStepAgent:
    model = LiteLLMModel(model_id=model_id)
    
    # Explicitly create tools
    web_tool = VisitWebpageTool()
    web_search_tool = DuckDuckGoSearchTool()
    agent = ToolCallingAgent(
        tools=[web_tool, web_search_tool],
        model=model,
        verbose=verbose,
        add_base_tools=False  # We're adding tools explicitly
    )
    return agent


def main():
    # Create the agent with debug-level logging
    agent = setup_simple_agent(verbose=True)
    # Run a simple query
    query = """
        Today's date is January 15, 2025. Who is the current CEO of Huggingface according
        to the news? Make sure this is up to date as of today, and cite both a webpage
        that contained this information, and a snippet of text from it which backs up
        this claim.
    """
    print(f"\nQuerying: {query}\n")
    
    # Run the agent and capture the result
    result = agent.run(query)
    
    # Print the final result
    print(f"\nFinal Result: {result}\n")

if __name__ == "__main__":
    main()