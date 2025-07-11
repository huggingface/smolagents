# Installation of smolagent

1. Install vllm
    1. Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
    2. Run 
    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

2. Install other python packages:
 
```bash
uv pip install flit
uv pip install .
flit install
```

3. Pass Hugging face key as an environment variable: run `export HF_TOKEN=your_huggingface_api_key_here`
You can also set mistral-AI API keys.

4. Compile the file:
```bash
make
make tests
```