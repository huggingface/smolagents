from smolagents import CodeAgent, VisitWebpageTool, InferenceClientModel

# hugging face token通过默认系统环境变量名HF_TOKEN获取

agent = CodeAgent(
    tools = [VisitWebpageTool()],
    model=InferenceClientModel(),
    # 将授权模块作为字符串列表传递给参数 additional_authorized_imports 来授权额外的导入
    additional_authorized_imports=["requests", "markdownify"],
    # 设置执行方式
    executor_type="docker",
)

result=agent.run("What was Abraham Lincoln's preferred pet?",return_full_result=True)
# return_full_result=False则返回整个log，是TEXT类型，只有设置为True才返回RunResult对象，才有steps等属性

print(result.steps)
