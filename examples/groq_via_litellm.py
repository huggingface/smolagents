"""Example: running a smolagents CodeAgent against Groq via LiteLLMModel.

Groq (https://groq.com/) is supported through the ``LiteLLMModel`` class by passing a
``groq/<model-id>`` ``model_id``. There is no dedicated Groq model class in smolagents.

Setup:
    pip install 'smolagents[litellm]'
    export GROQ_API_KEY=<your-key>     # get one at https://console.groq.com/keys

Why ``CodeAgent`` and not ``ToolCallingAgent``?
    ``ToolCallingAgent`` has known tool-call format incompatibilities with the Groq API
    (see https://github.com/huggingface/smolagents/issues/1119). With Groq, prefer
    ``CodeAgent`` — it asks the model to write Python that calls the tools, which Groq
    models handle reliably. ``flatten_messages_as_text=True`` is set automatically for
    ``groq/``-prefixed model ids, so you do not need to pass it.

The free Groq tier is rate-limited; we cap ``max_steps`` to keep the demo well within
the default per-minute quota.
"""

import os

from smolagents import CodeAgent, LiteLLMModel, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    # Replace with a real weather API call. The body just needs to return a string.
    return f"Sunny, 25°C in {city}"


@tool
def get_population(country: str) -> str:
    """Get the approximate population of a country.

    Args:
        country: Name of the country.
    """
    # Toy values for demo purposes; replace with a real lookup.
    table = {
        "france": "~68 million",
        "japan": "~125 million",
        "brazil": "~215 million",
    }
    return table.get(country.lower(), "unknown")


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys "
            "and export it before running this example."
        )

    model = LiteLLMModel(
        model_id="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,  # recommended for more consistent code generation
    )

    agent = CodeAgent(
        tools=[get_weather, get_population],
        model=model,
        max_steps=6,  # stay well within Groq's free-tier rate limits
    )
    answer = agent.run("What is the weather in Paris, and roughly how many people live in France?")
    print(answer)


if __name__ == "__main__":
    main()
