# Agent 简介

> [!TIP]
> 译者注：Agent 的业内术语是“智能体”。本译文将保留 agent，不作翻译，以带来更高效的阅读体验。(在中文为主的文章中，It's easier to 注意到英文。Attention Is All You Need!)

## 🤔 什么是 agent？

任何使用 AI 的高效系统都需要为 LLM 提供某种访问现实世界的方式：例如调用搜索工具获取外部信息，或者操作某些程序以完成任务。换句话说，LLM 应该具有 **_Agent 能力_**。Agent 程序是 LLM 通往外部世界的门户。

> [!TIP]
> AI agent 是 **LLM 输出控制工作流的程序**。

任何利用 LLM 的系统都会将 LLM 输出集成到代码中。LLM 输入对代码工作流的影响程度就是 LLM 在系统中的 agent 能力级别。

请注意，根据这个定义，"Agent" 不是一个离散的、非 0 即 1 的定义：相反，"Agent 能力" 是一个连续谱系，随着你在工作流中给予 LLM 更多或更少的权力而变化。

请参见下表中 agent 能力在不同系统中的变化：

| Agent 能力级别 | 描述                                           | 名称       | 示例模式                                           |
| ------------ | ---------------------------------------------- | ---------- | -------------------------------------------------- |
| ☆☆☆          | LLM 输出对程序流程没有影响                     | 简单处理器 | `process_llm_output(llm_response)`                 |
| ★☆☆          | LLM 输出决定 if/else 分支                      | 路由       | `if llm_decision(): path_a() else: path_b()`       |
| ★★☆          | LLM 输出决定函数执行                           | 工具调用者 | `run_function(llm_chosen_tool, llm_chosen_args)`   |
| ★★★          | LLM 输出控制迭代和程序继续                     | 多步 Agent | `while llm_should_continue(): execute_next_step()` |
| ★★★          | 一个 agent 工作流可以启动另一个 agent 工作流 | 多 Agent   | `if llm_trigger(): execute_agent()`                |

多步 agent 具有以下代码结构：

```python
memory = [user_defined_task]
while llm_should_continue(memory): # 这个循环是多步部分
    action = llm_get_next_action(memory) # 这是工具调用部分
    observations = execute_action(action)
    memory += [action, observations]
```

这个 agent 系统在一个循环中运行，每一步执行一个新动作（该动作可能涉及调用一些预定义的 *工具*，这些工具只是函数），直到其观察结果表明已达到解决给定任务的满意状态。以下是一个多步 agent 如何解决简单数学问题的示例：

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"/>
</div>

## ✅ 何时使用 agent / ⛔ 何时避免使用

当你需要 LLM 确定应用程序的工作流时，agent 很有用。但它们通常有些过度。问题是：我真的需要工作流的灵活性来有效解决手头的任务吗？
如果预定义的工作流经常不足，这意味着你需要更多的灵活性。
让我们举个例子：假设你正在开发一个处理冲浪旅行网站客户请求的应用程序。

你可以提前知道请求将属于 2 个类别之一（基于用户选择），并且你为这 2 种情况都有预定义的工作流。

1. 想要了解旅行信息？⇒ 给他们访问搜索栏以搜索你的知识库
2. 想与销售交谈？⇒ 让他们填写联系表单。

如果这个确定性工作流适合所有查询，那就直接编码吧！这将为你提供一个 100% 可靠的系统，没有让不可预测的 LLM 干扰你的工作流而引入错误的风险。为了简单和稳健起见，建议规范化不使用任何 agent 行为。

但如果工作流不能提前确定得那么好呢？

例如，用户想问：`"I can come on Monday, but I forgot my passport so risk being delayed to Wednesday, is it possible to take me and my stuff to surf on Tuesday morning, with a cancellation insurance?"` 这个问题涉及许多因素，可能上述预定的标准都不足以满足这个请求。

如果预定义的工作流经常不足，这意味着你需要更多的灵活性。

这就是 agent 设置发挥作用的地方。

在上面的例子中，你可以创建一个多步 agent，它可以访问天气 API 获取天气预报，Google Maps API 计算旅行距离，员工在线仪表板和你的知识库上的 RAG 系统。

直到最近，计算机程序还局限于预定义的工作流，试图通过堆积 if/else 分支来处理复杂性。它们专注于极其狭窄的任务，如"计算这些数字的总和"或"找到这个图中的最短路径"。但实际上，大多数现实生活中的任务，如我们上面的旅行示例，都不适合预定义的工作流。agent 系统为程序打开了现实世界任务的大门！

## 为什么选择 `smolagents`？

对于一些低级的 agent 用例，如链或路由器，你可以自己编写所有代码。这样会更好，因为它可以让你更好地控制和理解你的系统。

但一旦你开始追求更复杂的行为，比如让 LLM 调用函数（即"工具调用"）或让 LLM 运行 while 循环（"多步 agent"），一些抽象就变得必要：

- 对于工具调用，你需要解析 agent 的输出，因此这个输出需要一个预定义的格式，如"Thought: I should call tool 'get_weather'. Action: get_weather(Paris)."，你用预定义的函数解析它，并且给 LLM 的系统提示应该通知它这个格式。
- 对于 LLM 输出决定循环的多步 agent，你需要根据上次循环迭代中发生的情况给 LLM 不同的提示：所以你需要某种记忆能力。

看到了吗？通过这两个例子，我们已经发现需要一些项目来帮助我们：

- 当然，一个作为系统引擎的 LLM
- agent 可以访问的工具列表
- 从 LLM 输出中提取工具调用的解析器
- 与解析器同步的系统提示
- 记忆能力

但是等等，既然我们给 LLM 在决策中留出了空间，它们肯定会犯错误：所以我们需要错误日志记录和重试机制。

所有这些元素都需要紧密耦合才能形成一个功能良好的系统。这就是为什么我们决定需要制作基本构建块来让所有这些东西协同工作。

## 代码 agent

在多步 agent 中，每一步 LLM 都可以编写一个动作，形式为调用外部工具。编写这些动作的常见格式（由 Anthropic、OpenAI 等使用）通常是"将动作编写为工具名称和要使用的参数的 JSON，然后解析以知道要执行哪个工具以及使用哪些参数"的不同变体。

[多项](https://huggingface.co/papers/2402.01030) [研究](https://huggingface.co/papers/2411.01747) [论文](https://huggingface.co/papers/2401.00812) 表明，在代码中进行工具调用的 LLM 要好得多。

原因很简单，_我们专门设计了我们的代码语言，使其成为表达计算机执行动作的最佳方式_。如果 JSON 片段是更好的表达方式，JSON 将成为顶级编程语言，编程将变得非常困难。

下图取自 [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030)，说明了用代码编写动作的一些优势：

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

与 JSON 片段相比，用代码编写动作提供了更好的：

- **可组合性：** 你能像定义 python 函数一样，将 JSON 动作嵌套在一起，或定义一组 JSON 动作以供重用吗？
- **对象管理：** 你如何在 JSON 中存储像 `generate_image` 这样的动作的输出？
- **通用性：** 代码被构建为简单地表达任何你可以让计算机做的事情。
- **LLM 训练数据中的表示：** 大量高质量的代码动作已经包含在 LLM 的训练数据中，这意味着它们已经为此进行了训练！
