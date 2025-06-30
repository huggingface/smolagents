from smolagents import CodeAgent, InferenceClientModel, PlanningStep


def interrupt_after_plan(memory_step, agent):
    if isinstance(memory_step, PlanningStep):
        agent.interrupt()


agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[],
    planning_interval=100,
    step_callbacks=[interrupt_after_plan],
)
