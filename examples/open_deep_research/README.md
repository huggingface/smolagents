# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)! This agent attempts to replicate OpenAI's model and achieve similar performance on research tasks.

Read more about this implementation's goal and methods in our [blog post](https://huggingface.co/blog/open-deep-research).


This agent achieves **55% pass@1** on the GAIA validation set, compared to **67%** for the original Deep Research.

---

## Installation and Setup

To get started, follow the steps below:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/huggingface/smolagents.git
    cd smolagents/examples/open_deep_research
    ```

2. **Install required dependencies**:

    Run the following command to install the required dependencies from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3. **Install the development version of `smolagents`**:

    ```bash
    pip install smolagents[dev]
    ```

4. **Set up environment variables**:

    The script requires API keys to function:

    - **Hugging Face API Token**: You'll need a Hugging Face account and API token to run Open Deep Research.
        ```bash
        export HF_TOKEN="your_huggingface_token_here"
        ```
    
    - **OpenAI API Key**: You also need an OpenAI API key to run the script with OpenAI's models. You can obtain your key from OpenAI.
        ```bash
        export OPENAI_API_KEY="your_openai_api_key_here"
        ```
    
    - **SerpAPI Key**: Open Deep Research uses SerpAPI for web browsing functionality. 
        ```bash
        export SERPER_API_KEY="your_serpapi_key_here"
        ```

5. **Run the script**:

    Now you're ready to run the agent and ask your question! Simply run:

    ```bash
    python run.py --model-id "o1" "Your question here!"
    ```

    Replace `"Your question here!"` with the question you'd like the agent to answer. The agent will use the Hugging Face and OpenAI models to process and respond to your query.

---

## Notes

- Currently, this implementation only supports **OpenAI** models.
- Only **reasoning models** like `o1` are supported at this time. If you attempt to use a non-reasoning model, you may need to modify the script.

---
