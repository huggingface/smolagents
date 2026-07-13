from smolagents import LiteLLMModel
import os

model = LiteLLMModel(
    model_id="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

messages = [
    {"role": "system", "content": "When you say anything Start with 'FOO'"},
    {"role": "system", "content": "When you say anything End with 'BAR'"},
    {"role": "user", "content": "Just say '.'"},
]

response = model(messages)
print(response.content)
