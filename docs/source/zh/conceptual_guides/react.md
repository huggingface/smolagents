# 多步骤 agent 是如何工作的？

ReAct 框架（[Yao et al., 2022](https://huggingface.co/papers/2210.03629)）是目前构建 agent 的主要方法。

该名称基于两个词的组合："Reason" （推理）和 "Act" （行动）。实际上，遵循此架构的 agent 将根据需要尽可能多的步骤来解决其任务，每个步骤包括一个推理步骤，然后是一个行动步骤，在该步骤中，它制定工具调用，使其更接近解决手头的任务。

ReAct 过程涉及保留过去步骤的记忆。

> [!TIP]
> 阅读 [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) 博客文章以了解更多关于多步 agent 的信息。

以下是其工作原理的视频概述：

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

![ReAct agent 的框架](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

我们实现了两个版本的 ToolCallingAgent：
- [`ToolCallingAgent`] 在其输出中生成 JSON 格式的工具调用。
- [`CodeAgent`] 是一种新型的 ToolCallingAgent，它生成代码块形式的工具调用，这对于具有强大编码性能的 LLM 非常有效。
