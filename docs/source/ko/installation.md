# Installation Options

The `smolagents` library can be installed using pip. Here are the different installation methods and options available.

## Prerequisites
- Python 3.10 or newer
- Python package manager: [`pip`](https://pip.pypa.io/en/stable/) or [`uv`](https://docs.astral.sh/uv/)

## Virtual Environment

It's strongly recommended to install `smolagents` within a Python virtual environment.
Virtual environments isolate your project dependencies from other Python projects and your system Python installation,
preventing version conflicts and making package management more reliable.

<hfoptions id="virtual-environment">
<hfoption id="venv">

Using [`venv`](https://docs.python.org/3/library/venv.html):

```bash
python -m venv .venv
source .venv/bin/activate
```

</hfoption>
<hfoption id="uv">

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate
```

</hfoption>
</hfoptions>

## Basic Installation

Install `smolagents` core library with:

<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install smolagents
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install smolagents
```
</hfoption>
</hfoptions>

## Installation with Extras

`smolagents` provides several optional dependencies (extras) that can be installed based on your needs.
You can install these extras using the following syntax:
<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[extra1,extra2]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[extra1,extra2]"
```
</hfoption>
</hfoptions>

### Tools
These extras include various tools and integrations:
<hfoptions id="installation">
<hfoption id="pip">
- **toolkit**: Install a default set of tools for common tasks.
  ```bash
  pip install "smolagents[toolkit]"
  ```
- **mcp**: Add support for the Model Context Protocol (MCP) to integrate with external tools and services.
  ```bash
  pip install "smolagents[mcp]"
  ```
</hfoption>
<hfoption id="uv">
- **toolkit**: Install a default set of tools for common tasks.
  ```bash
  uv pip install "smolagents[toolkit]"
  ```
- **mcp**: Add support for the Model Context Protocol (MCP) to integrate with external tools and services.
  ```bash
  uv pip install "smolagents[mcp]"
  ```
</hfoption>
</hfoptions>

### Model Integration
These extras enable integration with various AI models and frameworks:
<hfoptions id="installation">
<hfoption id="pip">
- **openai**: Add support for OpenAI API models.
  ```bash
  pip install "smolagents[openai]"
  ```
- **transformers**: Enable Hugging Face Transformers models.
  ```bash
  pip install "smolagents[transformers]"
  ```
- **vllm**: Add VLLM support for efficient model inference.
  ```bash
  pip install "smolagents[vllm]"
  ```
- **mlx-lm**: Enable support for MLX-LM models.
  ```bash
  pip install "smolagents[mlx-lm]"
  ```
- **litellm**: Add LiteLLM support for lightweight model inference.
  ```bash
  pip install "smolagents[litellm]"
  ```
- **bedrock**: Enable support for AWS Bedrock models.
  ```bash
  pip install "smolagents[bedrock]"
  ```
</hfoption>
<hfoption id="uv">
- **openai**: Add support for OpenAI API models.
  ```bash
  uv pip install "smolagents[openai]"
  ```
- **transformers**: Enable Hugging Face Transformers models.
  ```bash
  uv pip install "smolagents[transformers]"
  ```
- **vllm**: Add VLLM support for efficient model inference.
  ```bash
  uv pip install "smolagents[vllm]"
  ```
- **mlx-lm**: Enable support for MLX-LM models.
  ```bash
  uv pip install "smolagents[mlx-lm]"
  ```
- **litellm**: Add LiteLLM support for lightweight model inference.
  ```bash
  uv pip install "smolagents[litellm]"
  ```
- **bedrock**: Enable support for AWS Bedrock models.
  ```bash
  uv pip install "smolagents[bedrock]"
  ```
</hfoption>
</hfoptions>

### Multimodal Capabilities
Extras for handling different types of media and input:
<hfoptions id="installation">
<hfoption id="pip">
- **vision**: Add support for image processing and computer vision tasks.
  ```bash
  pip install "smolagents[vision]"
  ```
- **audio**: Enable audio processing capabilities.
  ```bash
  pip install "smolagents[audio]"
  ```
</hfoption>
<hfoption id="uv">
- **vision**: Add support for image processing and computer vision tasks.
  ```bash
  uv pip install "smolagents[vision]"
  ```
- **audio**: Enable audio processing capabilities.
  ```bash
  uv pip install "smolagents[audio]"
  ```
</hfoption>
</hfoptions>

### Remote Execution
Extras for executing code remotely:
<hfoptions id="installation">
<hfoption id="pip">
- **docker**: Add support for executing code in Docker containers.
  ```bash
  pip install "smolagents[docker]"
  ```
- **e2b**: Enable E2B support for remote execution.
  ```bash
  pip install "smolagents[e2b]"
  ```
</hfoption>
<hfoption id="uv">
- **docker**: Add support for executing code in Docker containers.
  ```bash
  uv pip install "smolagents[docker]"
  ```
- **e2b**: Enable E2B support for remote execution.
  ```bash
  uv pip install "smolagents[e2b]"
  ```
</hfoption>
</hfoptions>

### Telemetry and User Interface
Extras for telemetry, monitoring and user interface components:
<hfoptions id="installation">
<hfoption id="pip">
- **telemetry**: Add support for monitoring and tracing.
  ```bash
  pip install "smolagents[telemetry]"
  ```
- **gradio**: Add support for interactive Gradio UI components.
  ```bash
  pip install "smolagents[gradio]"
  ```
</hfoption>
<hfoption id="uv">
- **telemetry**: Add support for monitoring and tracing.
  ```bash
  uv pip install "smolagents[telemetry]"
  ```
- **gradio**: Add support for interactive Gradio UI components.
  ```bash
  uv pip install "smolagents[gradio]"
  ```
</hfoption>
</hfoptions>

### Complete Installation
To install all available extras, you can use:
<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[all]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[all]"
```
</hfoption>
</hfoptions>

## Verifying Installation
After installation, you can verify that `smolagents` is installed correctly by running:
```python
import smolagents
print(smolagents.__version__)
```

## Next Steps
Once you have successfully installed `smolagents`, you can:
- Follow the [guided tour](./guided_tour) to learn the basics.
- Explore the [how-to guides](./examples/text_to_sql) for practical examples.
- Read the [conceptual guides](./conceptual_guides/intro_agents) for high-level explanations.
- Check out the [tutorials](./tutorials/building_good_agents) for in-depth tutorials on building agents.
- Explore the [API reference](./reference/index) for detailed information on classes and functions.
