# main.py
from multiprocessing import Manager, Process
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

def start_agent(agent_id: int, queue_dict: dict, tools: list[Tool], model: Model):
    """Start an agent as a separate process."""
    agent = CodeAgent(
        tools=tools,
        model=model,
        agent_id=agent_id,
        queue_dict=queue_dict,
    )
    agent.run()

if __name__ == "__main__":
    from smolagents import InferenceClientModel, WebSearchTool
    model = InferenceClientModel()
    tools = [WebSearchTool()]
    num_agents = 2
    with Manager() as manager:
        queue_dict = manager.dict()
        for i in range(num_agents):
            queue_dict[i] = manager.Queue()
        processes = []
        for i in range(num_agents):
            p = Process(target=start_agent, args=(i, queue_dict, tools, model))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()