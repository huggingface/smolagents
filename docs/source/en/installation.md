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
- **toolkit**: Install a default set of tools for common tasks.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[toolkit]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[toolkit]"
  ```
  </hfoption>
  </hfoptions>
- **mcp**: Add support for the Model Context Protocol (MCP) to integrate with external tools and services.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[mcp]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[mcp]"
  ```
  </hfoption>
  </hfoptions>

### Model Integration
These extras enable integration with various AI models and frameworks:
- **openai**: Add support for OpenAI API models.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[openai]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[openai]"
  ```
  </hfoption>
  </hfoptions>
- **transformers**: Enable Hugging Face Transformers models.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[transformers]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[transformers]"
  ```
  </hfoption>
  </hfoptions>
- **vllm**: Add VLLM support for efficient model inference.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[vllm]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[vllm]"
  ```
  </hfoption>
  </hfoptions>
- **mlx-lm**: Enable support for MLX-LM models.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[mlx-lm]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[mlx-lm]"
  ```
  </hfoption>
  </hfoptions>
- **litellm**: Add LiteLLM support for lightweight model inference.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[litellm]"
  ```
- **bedrock**: Enable support for AWS Bedrock models.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[bedrock]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[bedrock]"
  ```
  </hfoption>
  </hfoptions>

### Multimodal Capabilities
Extras for handling different types of media and input:
- **vision**: Add support for image processing and computer vision tasks.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[vision]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[vision]"
  ```
  </hfoption>
  </hfoptions>
- **audio**: Enable audio processing capabilities.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[audio]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[audio]"
  ```
  </hfoption>
  </hfoptions>

### Remote Execution
Extras for executing code remotely:
- **docker**: Add support for executing code in Docker containers.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[docker]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[docker]"
  ```
  </hfoption>
  </hfoptions>
- **e2b**: Enable E2B support for remote execution.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[e2b]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[e2b]"
  ```
  </hfoption>
  </hfoptions>

### Telemetry and User Interface
Extras for telemetry, monitoring and user interface components:
- **telemetry**: Add support for monitoring and tracing.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[telemetry]"
  ```
  </hfoption>
  <hfoption id="uv">
  ```bash
  uv pip install "smolagents[telemetry]"
  ```
  </hfoption>
  </hfoptions>
- **gradio**: Add support for interactive Gradio UI components.
  <hfoptions id="installation">
  <hfoption id="pip">
  ```bash
  pip install "smolagents[gradio]"
  ```
  </hfoption>
  <hfoption id="uv">
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
