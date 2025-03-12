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

The agent uses the `GoogleSearchTool` for web search, which requires an environment variable with the corresponding API key, based on the selected provider:
- `SERPAPI_API_KEY` for SerpApi: [Sign up here to get a key](https://serpapi.com/users/sign_up)
- `SERPER_API_KEY` for Serper: [Sign up here to get a key](https://serper.dev/signup)

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

## Full reproducibility of results
> [!WARNING]
> The data used in our submissions to GAIA was augmented in this way:
> -  For each single-page .pdf or .xls file, it was opened in a file reader (MacOS Sonoma Numbers or Preview), and a ".png" screenshot was taken and added to the folder.
> - Then for any file used in a question, the file loading system checks if there is a ".png" extension version of the file, and loads it instead of the original if it exists.
> This process was done manually but could be automatized.
