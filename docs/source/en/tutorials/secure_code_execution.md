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
# Secure code execution

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

### Code agents

[Multiple](https://huggingface.co/papers/2402.01030) [research](https://huggingface.co/papers/2411.01747) [papers](https://huggingface.co/papers/2401.00812) have shown that having the LLM write its actions (the tool calls) in code is much better than the current standard format for tool calling, which is across the industry different shades of "writing actions as a JSON of tools names and arguments to use".

Why is code better? Well, because we crafted our code languages specifically to be great at expressing actions performed by a computer. If JSON snippets was a better way, this package would have been written in JSON snippets and the devil would be laughing at us.

Code is just a better way to express actions on a computer. It has better:
- **Composability:** could you nest JSON actions within each other, or define a set of JSON actions to re-use later, the same way you could just define a python function?
- **Object management:** how do you store the output of an action like `generate_image` in JSON?
- **Generality:** code is built to express simply anything you can do have a computer do.
- **Representation in LLM training corpus:** why not leverage this benediction of the sky that plenty of quality actions have already been included in LLM training corpus?

This is illustrated on the figure below, taken from [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

This is why we put emphasis on proposing code agents, in this case python agents, which meant putting higher effort on building secure python interpreters.

### Local python interpreter

By default, the `CodeAgent` runs LLM-generated code in your environment.
This execution is not done by the vanilla Python interpreter: we've re-built a more secure `LocalPythonInterpreter` from the ground up.
This interpreter is designed for security by:
 - Restricting the imports to a list explicitly passed by the user
 - Capping the number of operations to prevent infinite loops and resource bloating.
 - Will not perform any operation that's not pre-defined.

We've used this on many use cases, without ever observing any damage to the environment. 

However this solution is not watertight: one could imagine occasions where LLMs fine-tuned for malignant actions could still hurt your environment. For instance if you've allowed an innocuous package like `Pillow` to process images, the LLM could generate thousands of saves of images to bloat your hard drive.
It's certainly not likely if you've chosen the LLM engine yourself, but it could happen.

So if you want to be extra cautious, you can use the remote code execution option described below.

### E2B code executor

For maximum security, you can use our integration with E2B to run code in a sandboxed environment. This is a remote execution service that runs your code in an isolated container, making it impossible for the code to affect your local environment.

For this, you will need to setup your E2B account and set your `E2B_API_KEY` in your environment variables. Head to [E2B's quickstart documentation](https://e2b.dev/docs/quickstart) for more information.

Then you can install it with `pip install "smolagents[e2b]"`.

Now you're set!

To set the code executor to E2B, simply pass the flag `use_e2b_executor=True` when initializing your `CodeAgent`.
Note that you should add all the tool's dependencies in `additional_authorized_imports`, so that the executor installs them.

```py
from smolagents import CodeAgent, VisitWebpageTool, HfApiModel
agent = CodeAgent(
    tools = [VisitWebpageTool()],
    model=HfApiModel(),
    additional_authorized_imports=["requests", "markdownify"],
    use_e2b_executor=True
)

agent.run("What was Abraham Lincoln's preferred pet?")
```

E2B code execution is not compatible with multi-agents at the moment - because having an agent call in a code blob that should be executed remotely is a mess. But we're working on adding it!
