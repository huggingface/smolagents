<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# `smolagents`

This library is the simplest framework out there to build powerful agents! By the way, wtf are "agents"? We provide our definition [in this page](conceptual_guides/intro_agents), where you'll also find tips for when to use them or not (spoilers: you'll often be better off without agents).

This library offers:

✨ **Simplicity**: the logic for agents fits in ~thousand lines of code. We kept abstractions to their minimal shape above raw code!

🌐 **Support for any LLM**: it supports models hosted on the Hub loaded in their `transformers` version or through our inference API, but also models from OpenAI, Anthropic... it's really easy to power an agent with any LLM.

🧑‍💻 **First-class support for Code Agents**, i.e. agents that write their actions in code (as opposed to "agents being used to write code"), [read more here](tutorials/secure_code_execution).

🤗 **Hub integrations**: you can share and load tools to/from the Hub, and more is to come!

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guided tour</div>
      <p class="text-gray-700">Learn the basics and become familiar with using Agents. Start here if you are using Agents for the first time!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">How-to guides</div>
      <p class="text-gray-700">Practical guides to help you achieve a specific goal: create an agent to generate and test SQL queries!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Conceptual guides</div>
      <p class="text-gray-700">High-level explanations for building a better understanding of important topics.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
      <p class="text-gray-700">Horizontal tutorials that cover important aspects of building agents.</p>
    </a>
  </div>
</div>
