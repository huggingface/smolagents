# `smolagents`

这是构建强大 agent 的最简单框架！顺便问一下，什么是 "agent"？我们在[此页面](conceptual_guides/intro_agents)提供了我们的定义，您还可以找到关于何时使用或不使用它们的建议（剧透：通常不使用 agent 会更好）。

> [!TIP]
> 译者注：Agent 的业内术语是“智能体”。本译文将保留 agent，不作翻译，以带来更高效的阅读体验。(在中文为主的文章中，It's easier to 注意到英文。Attention Is All You Need!)

本库提供：

✨ **简洁性**：Agent 逻辑仅需约千行代码。我们将抽象保持在原始代码之上的最小形态！

🌐 **支持任何 LLM**：支持通过 Hub 托管的模型，使用其 `transformers` 版本或通过我们的推理 API 加载，也支持 OpenAI、Anthropic 等模型。使用任何 LLM 为 agent 提供动力都非常容易。

🧑‍💻 **一流的代码 agent 支持**，即编写代码作为其操作的 agent（与"用于编写代码的 agent"相对），[在此了解更多](tutorials/secure_code_execution)。

🤗 **Hub 集成**：您可以在 Hub 上共享和加载工具，更多功能即将推出！

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">导览</div>
      <p class="text-gray-700">学习基础知识并熟悉使用 agent。如果您是第一次使用 agent，请从这里开始！</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">操作指南</div>
      <p class="text-gray-700">实用指南，帮助您实现特定目标：创建一个生成和测试 SQL 查询的 agent！</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">概念指南</div>
      <p class="text-gray-700">高级解释，帮助您更好地理解重要主题。</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">教程</div>
      <p class="text-gray-700">涵盖构建 agent 重要方面的横向教程。</p>
    </a>
  </div>
</div>
