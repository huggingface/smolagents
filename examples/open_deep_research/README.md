# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)!

Read more about this implementation's goal and methods [in our blog post](https://huggingface.co/blog/open-deep-research).

This agent achieves 55% pass@1 on GAIA validation set, vs 67% for Deep Research.

## Setup

### Installation

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install -e ../../.[dev]
```

### Environment variables
Create a `.env` file in the `examples/open_deep_research` directory of the project with the following content:
```bash
HF_TOKEN="your_huggingface_token"
SERPAPI_API_KEY="your_serpapi_key"
OPENAI_API_KEY="your_openai_key"
LINKUP_API_KEY="your_linkup_key"
```
### Web search using GoogleSearchTool
The agent uses the `GoogleSearchTool` for web search, which requires an environment variable with the corresponding API key, based on the selected provider:
- `SERPAPI_API_KEY` for SerpApi: [Sign up here to get a key](https://serpapi.com/users/sign_up)
- `SERPER_API_KEY` for Serper: [Sign up here to get a key](https://serper.dev/signup)

### Websearch using LinkupSearchTool
If the agent uses the `LinkupSearchTool` for web search, which requires an environment variable with the corresponding API key:
- `LINKUP_API_KEY` for Linkup: [Sign up here to get a key](https://app.linkup.so/sign-up)

Depending on the model you want to use, you may need to set environment variables.
For example, to use the default `o1` model, you need to set the `OPENAI_API_KEY` environment variable.
[Sign up here to get a key](https://platform.openai.com/signup).

> [!WARNING]
> The use of the default `o1` model is restricted to tier-3 access: https://help.openai.com/en/articles/10362446-api-access-to-o1-and-o3-mini

## Usage

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```
