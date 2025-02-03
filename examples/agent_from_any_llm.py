from typing import Optional

from smolagents import HfApiModel, LiteLLMModel, TransformersModel, LlamaCppModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent


# Choose which inference type to use!

available_inferences = ["hf_api", "transformers", "ollama", "litellm", "llama_cpp"]
chosen_inference = "llama_cpp"

print(f"Chose model: '{chosen_inference}'")

if chosen_inference == "hf_api":
    model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct")

elif chosen_inference == "transformers":
    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)

elif chosen_inference == "ollama":
    model = LiteLLMModel(
        model_id="ollama_chat/llama3.2",
        api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
        api_key="your-api-key",  # replace with API key if necessary
        num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
    )

elif chosen_inference == "litellm":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = LiteLLMModel(model_id="gpt-4o")

elif chosen_inference == "llama_cpp":
    model = LlamaCppModel(
        repo_id="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf",
        n_ctx=8192,
    )

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


agent = ToolCallingAgent(model=model, tools=[get_weather], max_steps=1)
print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

agent = CodeAgent(model=model, tools=[get_weather])
print("CodeAgent:", agent.run("What's the weather like in Paris?"))
