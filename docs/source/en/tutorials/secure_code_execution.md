# Secure code execution

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

### Code agents

[Multiple](https://huggingface.co/papers/2402.01030) [research](https://huggingface.co/papers/2411.01747) [papers](https://huggingface.co/papers/2401.00812) have shown that having the LLM write its actions (the tool calls) in code is much better than the current standard format for tool calling, which is across the industry different shades of "writing actions as a JSON of tools names and arguments to use".

Why is code better? Well, because we crafted our code languages specifically to be great at expressing actions performed by a computer. If JSON snippets were a better way, this package would have been written in JSON snippets and the devil would be laughing at us.

Code is just a better way to express actions on a computer. It has better:
- **Composability:** could you nest JSON actions within each other, or define a set of JSON actions to re-use later, the same way you could just define a python function?
- **Object management:** how do you store the output of an action like `generate_image` in JSON?
- **Generality:** code is built to express simply anything you can have a computer do.
- **Representation in LLM training corpus:** why not leverage this benediction of the sky that plenty of quality actions have already been included in LLM training corpus?

This is illustrated on the figure below, taken from [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

This is why we put emphasis on proposing code agents, in this case python agents, which meant putting higher effort on building secure python interpreters.

### Local code execution??

By default, the `CodeAgent` runs LLM-generated code in your environment.

This is inherently risky, LLM-generated code could be harmful to your environment.

Malicious code execution can occur in several ways:
- **Plain LLM error:** LLMs are still far from perfect and may unintentionally generate harmful commands while attempting to be helpful. While this risk is low, instances have been observed where an LLM attempted to execute potentially dangerous code.  
- **Supply chain attack:** Running an untrusted or compromised LLM could expose a system to harmful code generation. While this risk is extremely low when using well-known models on secure inference infrastructure, it remains a theoretical possibility.  
- **Prompt injection:** an agent browsing the web could arrive on a malicious website that contains harmful instructions, thus injecting an attack into the agent's memory
- **Exploitation of publicly accessible agents:** Agents exposed to the public can be misused by malicious actors to execute harmful code. Attackers may craft adversarial inputs to exploit the agent's execution capabilities, leading to unintended consequences.
Once malicious code is executed, whether accidentally or intentionally, it can damage the file system, exploit local or cloud-based resources, abuse API services, and even compromise network security.

One could argue that on the [spectrum of agency](../conceptual_guides/intro_agents), code agents give much higher agency to the LLM on your system than other less agentic setups: this goes hand-in-hand with higher risk.

So you need to be very mindful of security.

To improve safety, we propose a range of measures that propose elevated levels of security, at a higher setup cost.

We advise you to keep in mind that no solution will be 100% safe.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/code_execution_safety_diagram.png">

### Our local Python executor

To add a first layer of security, code execution in `smolagents` is not performed by the vanilla Python interpreter.
We have re-built a more secure `LocalPythonExecutor` from the ground up.

To be precise, this interpreter works by loading the Abstract Syntax Tree (AST) from your Code and executes it operation by operation, making sure to always follow certain rules:
- By default, imports are disallowed unless they have been explicitly added to an authorization list by the user.
- Furthermore, access to submodules is disabled by default, and each must be explicitly authorized in the import list as well, or you can pass for instance `numpy.*` to allow both `numpy` and all its subpackags, like `numpy.random` or `numpy.a.b`.
   - Note that some seemingly innocuous packages like `random` can give access to potentially harmful submodules, as in `random._os`.
- The total count of elementary operations processed is capped to prevent infinite loops and resource bloating.
- Any operation that has not been explicitly defined in our custom interpreter will raise an error.

You could try these safeguards as follows:

```py
from smolagents.local_python_executor import LocalPythonExecutor

# Set up custom executor, authorize package "numpy"
custom_executor = LocalPythonExecutor(["numpy"])

# Utilisty for pretty printing errors
def run_capture_exception(command: str):
    try:
        custom_executor(harmful_command)
    except Exception as e:
        print("ERROR:\n", e)

# Undefined command just do not work
harmful_command="!echo Bad command"
run_capture_exception(harmful_command)
# >>> ERROR: invalid syntax (<unknown>, line 1)


# Imports like os will not be performed unless explicitly added to `additional_authorized_imports`
harmful_command="import os; exit_code = os.system('echo Bad command')"
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'import os' due to: InterpreterError: Import of os is not allowed. Authorized imports are: ['statistics', 'numpy', 'itertools', 'time', 'queue', 'collections', 'math', 'random', 're', 'datetime', 'stat', 'unicodedata']

# Even in authorized imports, potentially harmful packages will not be imported
harmful_command="import random; random._os.system('echo Bad command')"
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'random._os.system('echo Bad command')' due to: InterpreterError: Forbidden access to module: os

# Infinite loop are interrupted after N operations
harmful_command="""
while True:
    pass
"""
run_capture_exception(harmful_command)
# >>> ERROR: Code execution failed at line 'while True: pass' due to: InterpreterError: Maximum number of 1000000 iterations in While loop exceeded
```

These safeguards make out interpreter is safer.
We have used it on a diversity of use cases, without ever observing any damage to the environment.

> [!WARNING]
> It's important to understand that no local python sandbox can ever be completely secure. While our interpreter provides significant safety improvements over the standard Python interpreter, it is still possible for a determined attacker or a fine-tuned malicious LLM to find vulnerabilities and potentially harm your environment. 
> 
> For example, if you've allowed packages like `Pillow` to process images, the LLM could generate code that creates thousands of large image files to fill your hard drive. Other advanced escape techniques might exploit deeper vulnerabilities in authorized packages.
> 
> Running LLM-generated code in your local environment always carries some inherent risk. The only way to run LLM-generated code with truly robust security isolation is to use remote execution options like E2B or Docker, as detailed below.

The risk of a malicious attack is low when using well-known LLMs from trusted inference providers, but it is not zero.
For high-security applications or when using less trusted models, you should consider using a remote execution sandbox.

## Sandbox approaches for secure code execution

When working with AI agents that execute code, security is paramount. There are two main approaches to sandboxing code execution in smolagents, each with different security properties and capabilities:


![Sandbox approaches comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/sandboxed_execution.png)

1. **Running individual code snippets in a sandbox**: This approach (left side of diagram) only executes the agent-generated Python code snippets in a sandbox while keeping the rest of the agentic system in your local environment. It's simpler to set up using `executor_type="e2b"` or `executor_type="docker"`, but it doesn't support multi-agents and still requires passing state data between your environment and the sandbox.

2. **Running the entire agentic system in a sandbox**: This approach (right side of diagram) runs the entire agentic system, including the agent, model, and tools, within a sandbox environment. This provides better isolation but requires more manual setup and may require passing sensitive credentials (like API keys) to the sandbox environment.

This guide describes how to set up and use both types of sandbox approaches for your agent applications.

### E2B setup

#### Installation

1. Create an E2B account at [e2b.dev](https://e2b.dev)
2. Install the required packages:
```bash
pip install 'smolagents[e2b]'
```

#### Running your agent in E2B: quick start

We provide a simple way to use an E2B Sandbox: simply add `executor_type="e2b"` to the agent initialization, as follows:

```py
from smolagents import InferenceClientModel, CodeAgent

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="e2b") as agent:
    agent.run("Can you give me the 100th Fibonacci number?")
```

> [!TIP]
> Using the agent as a context manager (with the `with` statement) ensures that the E2B sandbox is cleaned up immediately after the agent completes its task.
> Alternatively, you can manually call the agent's `cleanup()` method.

This solution send the agent state to the server at the start of each `agent.run()`.
Then the models are called from the local environment, but the generated code will be sent to the sandbox for execution, and only the output will be returned.

This is illustrated in the figure below.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/sandboxed_execution.png" alt="sandboxed code execution" width=60% max-width=500px>
</p>

However, since any call to a [managed agent](../examples/multiagents) would require model calls, since we do not transfer secrets to the remote sandbox, the model call would lack credentials.
Hence this solution does not work (yet) with more complicated multi-agent setups.

#### Running your agent in E2B: multi-agents

To use multi-agents in an E2B sandbox, you need to run your agents completely from within E2B.

Here is how to do it:

```python
from e2b_code_interpreter import Sandbox
import os

# Create the sandbox
sandbox = Sandbox()

# Install required packages
sandbox.commands.run("pip install smolagents")

def run_code_raise_errors(sandbox, code: str, verbose: bool = False) -> str:
    execution = sandbox.run_code(
        code,
        envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])

# Define your agent application
agent_code = """
import os
from smolagents import CodeAgent, InferenceClientModel

# Initialize the agents
agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[],
    name="coder_agent",
    description="This agent takes care of your difficult algorithmic problems using code."
)

manager_agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[],
    managed_agents=[agent],
)

# Run the agent
response = manager_agent.run("What's the 20th Fibonacci number?")
print(response)
"""

# Run the agent code in the sandbox
execution_logs = run_code_raise_errors(sandbox, agent_code)
print(execution_logs)
```

### Docker setup

#### Installation

1. [Install Docker on your system](https://docs.docker.com/get-started/get-docker/)
2. Install the required packages:
```bash
pip install 'smolagents[docker]'
```

#### Running your agent in Docker: quick start

Similar to the E2B Sandbox above, to quickly get started with Docker, simply add `executor_type="docker"` to the agent initialization, like:

```py
from smolagents import InferenceClientModel, CodeAgent

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="docker") as agent:
    agent.run("Can you give me the 100th Fibonacci number?")
```

> [!TIP]
> Using the agent as a context manager (with the `with` statement) ensures that the Docker container is cleaned immediately after the agent completes its task.
> Alternatively, you can manually call the agent's `cleanup()` method.

#### Advanced docker usage

If you want to run multi-agent systems in Docker, you'll need to setup a custom interpreter in a sandbox.

Here is how to setup the a Dockerfile:

```dockerfile
FROM python:3.10-bullseye

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir smolagents && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Run with limited privileges
USER nobody

# Default command
CMD ["python", "-c", "print('Container ready')"]
```

Create a sandbox manager to run code:

```python
import docker
import os
from typing import Optional

class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def create_container(self):
        try:
            image, build_logs = self.client.images.build(
                path=".",
                tag="agent-sandbox",
                rm=True,
                forcerm=True,
                buildargs={},
                # decode=True
            )
        except docker.errors.BuildError as e:
            print("Build error logs:")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            raise

        # Create container with security constraints and proper logging
        self.container = self.client.containers.run(
            "agent-sandbox",
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            tty=True,
            mem_limit="512m",
            cpu_quota=50000,
            pids_limit=100,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str) -> Optional[str]:
        if not self.container:
            self.create_container()

        # Execute code in container
        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody"
        )

        # Collect all output
        return exec_result.output.decode() if exec_result.output else None


    def cleanup(self):
        if self.container:
            try:
                self.container.stop()
            except docker.errors.NotFound:
                # Container already removed, this is expected
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.container = None  # Clear the reference

# Example usage:
sandbox = DockerSandbox()

try:
    # Define your agent code
    agent_code = """
import os
from smolagents import CodeAgent, InferenceClientModel

# Initialize the agent
agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[]
)

# Run the agent
response = agent.run("What's the 20th Fibonacci number?")
print(response)
"""

    # Run the code in the sandbox
    output = sandbox.run_code(agent_code)
    print(output)

finally:
    sandbox.cleanup()
```

### WebAssembly setup

WebAssembly (Wasm) is a binary instruction format that allows code to be run in a safe, sandboxed environment.
It is designed to be fast, efficient, and secure, making it an excellent choice for executing potentially untrusted code.

The `WasmExecutor` uses [Pyodide](https://pyodide.org/) and [Deno](https://docs.deno.com/).

#### Installation

1. [Install Deno on your system](https://docs.deno.com/runtime/getting_started/installation/)

#### Running your agent in WebAssembly: quick start

Simply pass `executor_type="wasm"` to the agent initialization, like:
```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(model=InferenceClientModel(), tools=[], executor_type="wasm")

agent.run("Can you give me the 100th Fibonacci number?")
```

### Best practices for sandboxes

These key practices apply to both E2B and Docker sandboxes:

- Resource management
  - Set memory and CPU limits
  - Implement execution timeouts
  - Monitor resource usage
- Security
  - Run with minimal privileges
  - Disable unnecessary network access
  - Use environment variables for secrets
- Environment
  - Keep dependencies minimal
  - Use fixed package versions
  - If you use base images, update them regularly

- Cleanup
  - Always ensure proper cleanup of resources, especially for Docker containers, to avoid having dangling containers eating up resources.

✨ By following these practices and implementing proper cleanup procedures, you can ensure your agent runs safely and efficiently in a sandboxed environment.

## Comparing security approaches

As illustrated in the diagram earlier, both sandboxing approaches have different security implications:

### Approach 1: Running just the code snippets in a sandbox
- **Pros**: 
  - Easier to set up with a simple parameter (`executor_type="e2b"` or `executor_type="docker"`)
  - No need to transfer API keys to the sandbox
  - Better protection for your local environment
- **Cons**:
  - Doesn't support multi-agents (managed agents)
  - Still requires transferring state between your environment and the sandbox
  - Limited to specific code execution

### Approach 2: Running the entire agentic system in a sandbox
- **Pros**:
  - Supports multi-agents
  - Complete isolation of the entire agent system
  - More flexible for complex agent architectures
- **Cons**:
  - Requires more manual setup
  - May require transferring sensitive API keys to the sandbox
  - Potentially higher latency due to more complex operations

Choose the approach that best balances your security needs with your application's requirements. For most applications with simpler agent architectures, Approach 1 provides a good balance of security and ease of use. For more complex multi-agent systems where you need full isolation, Approach 2, while more involved to set up, offers better security guarantees.
