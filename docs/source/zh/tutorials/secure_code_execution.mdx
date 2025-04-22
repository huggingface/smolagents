# 安全代码执行

[[open-in-colab]]

> [!TIP]
> 如果你是第一次构建 agent，请先阅读 [agent 介绍](../conceptual_guides/intro_agents) 和 [smolagents 导览](../guided_tour)。

### 代码智能体

[多项](https://huggingface.co/papers/2402.01030) [研究](https://huggingface.co/papers/2411.01747) [表明](https://huggingface.co/papers/2401.00812)，让大语言模型用代码编写其动作（工具调用）比当前标准的工具调用格式要好得多，目前行业标准是 "将动作写成包含工具名称和参数的 JSON" 的各种变体。

为什么代码更好？因为我们专门为计算机执行的动作而设计编程语言。如果 JSON 片段是更好的方式，那么这个工具包就应该是用 JSON 片段编写的，魔鬼就会嘲笑我们。

代码就是表达计算机动作的更好方式。它具有更好的：
- **组合性**：你能像定义 Python 函数那样，在 JSON 动作中嵌套其他 JSON 动作，或者定义一组 JSON 动作以便以后重用吗？
- **对象管理**：你如何在 JSON 中存储像 `generate_image` 这样的动作的输出？
- **通用性**：代码是为了简单地表达任何可以让计算机做的事情而构建的。
- **在 LLM 训练语料库中的表示**：天赐良机，为什么不利用已经包含在 LLM 训练语料库中的大量高质量动作呢？

下图展示了这一点，取自 [可执行代码动作引出更好的 LLM 智能体](https://huggingface.co/papers/2402.01030)。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

这就是为什么我们强调提出代码智能体，在本例中是 Python 智能体，这意味着我们要在构建安全的 Python 解释器上投入更多精力。

### 本地 Python 解释器

默认情况下，`CodeAgent` 会在你的环境中运行 LLM 生成的代码。
这个执行不是由普通的 Python 解释器完成的：我们从零开始重新构建了一个更安全的 `LocalPythonExecutor`。
这个解释器通过以下方式设计以确保安全：
  - 将导入限制为用户显式传递的列表
  - 限制操作次数以防止无限循环和资源膨胀
  - 不会执行任何未预定义的操作

我们已经在许多用例中使用了这个解释器，从未观察到对环境造成任何损害。

然而，这个解决方案并不是万无一失的：可以想象，如果 LLM 被微调用于恶意操作，仍然可能损害你的环境。例如，如果你允许像 `Pillow` 这样无害的包处理图像，LLM 可能会生成数千张图像保存以膨胀你的硬盘。
如果你自己选择了 LLM 引擎，这当然不太可能，但它可能会发生。

所以如果你想格外谨慎，可以使用下面描述的远程代码执行选项。

### E2B 代码执行器

为了最大程度的安全性，你可以使用我们与 E2B 的集成在沙盒环境中运行代码。这是一个远程执行服务，可以在隔离的容器中运行你的代码，使代码无法影响你的本地环境。

为此，你需要设置你的 E2B 账户并在环境变量中设置 `E2B_API_KEY`。请前往 [E2B 快速入门文档](https://e2b.dev/docs/quickstart) 了解更多信息。

然后你可以通过 `pip install e2b-code-interpreter python-dotenv` 安装它。

现在你已经准备好了！

要将代码执行器设置为 E2B，只需在初始化 `CodeAgent` 时传递标志 `executor_type="e2b"`。
请注意，你应该将所有工具的依赖项添加到 `additional_authorized_imports` 中，以便执行器安装它们。

```py
from smolagents import CodeAgent, VisitWebpageTool, InferenceClientModel
agent = CodeAgent(
    tools = [VisitWebpageTool()],
    model=InferenceClientModel(),
    additional_authorized_imports=["requests", "markdownify"],
    executor_type="e2b"
)

agent.run("What was Abraham Lincoln's preferred pet?")
```

目前 E2B 代码执行暂不兼容多 agent——因为把 agent 调用放在应该在远程执行的代码块里，是非常混乱的。但我们正在努力做到这件事！
