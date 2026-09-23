## 参考
https://github.com/huggingface/smolagents/blob/main/docs/source/zh/conceptual_guides/intro_agents.md
https://github.com/huggingface/smolagents/blob/main/docs/source/zh/conceptual_guides/react.md

## 内容

工作流：为了完成某个目标，系统按照一定的步骤、顺序和决策规则处理输入，最终得到输出的过程。

AI agent 是由 LLM 输出影响或控制工作流的程序。Agent 运行时把 LLM 输出解释为行动或决策，执行后将结果反馈给 LLM，不断推进任务，直到产生最终答案或达到终止条件。

Agent 可以只让 LLM 决定进入哪个预定义分支（路由）；也可以让 LLM 决定调用什么工具以及使用什么参数（工具调用者）；还可以把行动产生的观察结果写入记忆，供 LLM 继续决策，直到任务结束（多步 agent ）；甚至可以调用预先配置好的其他 Agent，把对方的完整工作流嵌入自己的工作流（多 agent ）。

多步 agent 的大体流程：
```
memory = [user_defined_task]
while llm_should_continue(memory): # 这个循环是多步部分
    action = llm_get_next_action(memory) # 这是工具调用部分
    observations = execute_action(action)
    memory += [action, observations]
```

如果任务的步骤、顺序和分支可以提前确定，就优先使用普通代码完成确定性工作流；只有当工作流难以预先穷举，需要根据情况灵活选择下一步时，我们才考虑引入 agent。

要做成一个比较高级的 agent，如多步 agent，我们需要 LLM、agent 能调用的工具列表、把 LLM 输出转化为操作的命令的解析器、让LLM按照解析器格式输出内容的 prompt、记忆能力、错误日志和重试机制。

Agent 可以使用不同格式表达动作，例如 JSON 工具调用或代码。`smolagents` 提供了 `CodeAgent`，让 LLM 使用 Python 代码表达动作。代码可以利用变量、条件、循环和函数，将多个工具调用及中间处理组合成一个更完整的行动；工具返回结果也可以作为对象保存并继续使用。代码本身是为表达计算机操作而设计的，而且 LLM 的训练数据中包含大量高质量代码，因此代码通常是比自然语言或单个 JSON 调用更有表现力的行动格式。

单步 agent 完成一次 LLM 决策和行动后，工作流不会再把行动结果反馈给 LLM 进行下一轮决策。

ReAct 框架是多步骤 agent 的一种实现方式，它的大致流程可以理解为：每一轮中，LLM 根据任务和已有记忆进行推理并提出行动；Agent 运行时解析、验证并执行行动，得到环境返回的观察结果；随后将行动和观察写入记忆。下一轮，运行时根据更新后的记忆重新构造模型输入，让 LLM 决定新的行动或提交最终答案。这个过程持续到模型返回最终答案，或者达到最大步数等终止条件。

smolagents 基于 ReAct 框架提供了两种主要的行动表达方式：ToolCallingAgent 使用结构化工具调用，CodeAgent 使用 Python 代码表达和组合行动。