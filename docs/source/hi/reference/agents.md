# Agents

<Tip warning={true}>

Smolagents एक experimental API है जो किसी भी समय बदल सकता है। एजेंट्स द्वारा लौटाए गए परिणाम भिन्न हो सकते हैं क्योंकि APIs या underlying मॉडल बदलने की संभावना रखते हैं।

</Tip>

Agents और tools के बारे में अधिक जानने के लिए [introductory guide](../index) पढ़ना सुनिश्चित करें। 
यह पेज underlying क्लासेज के लिए API docs को शामिल करता है।

## Agents

हमारे एजेंट्स [`MultiStepAgent`] से इनहेरिट करते हैं, जिसका अर्थ है कि वे कई चरणों में कार्य कर सकते हैं, प्रत्येक चरण में एक विचार, फिर एक टूल कॉल और एक्जीक्यूशन शामिल होता है। [इस कॉन्सेप्चुअल गाइड](../conceptual_guides/react) में अधिक पढ़ें।

हम मुख्य [`Agent`] क्लास पर आधारित दो प्रकार के एजेंट्स प्रदान करते हैं।
  - [`CodeAgent`] डिफ़ॉल्ट एजेंट है, यह अपने टूल कॉल्स को Python कोड में लिखता है।
  - [`ToolCallingAgent`] अपने टूल कॉल्स को JSON में लिखता है।

दोनों को इनिशियलाइजेशन पर `model` और टूल्स की सूची `tools` आर्गुमेंट्स की आवश्यकता होती है।

### Agents की क्लासेज

[[autodoc]] MultiStepAgent

[[autodoc]] CodeAgent

[[autodoc]] ToolCallingAgent

### stream_to_gradio

[[autodoc]] stream_to_gradio

### GradioUI

[[autodoc]] GradioUI

## मॉडल्स

आप स्वतंत्र रूप से अपने स्वयं के मॉडल बना सकते हैं और उनका उपयोग कर सकते हैं।

आप अपने एजेंट के लिए कोई भी `model` कॉल करने योग्य उपयोग कर सकते हैं, जब तक कि:
1. यह अपने इनपुट `messages` के लिए [messages format](./chat_templating) (`List[Dict[str, str]]`) का पालन करता है, और यह एक `str` लौटाता है।
2. यह आर्गुमेंट `stop_sequences` में पास किए गए सीक्वेंस से *पहले* आउटपुट जनरेट करना बंद कर देता है।

अपने LLM को परिभाषित करने के लिए, आप एक `custom_model` मेथड बना सकते हैं जो [messages](./chat_templating) की एक सूची स्वीकार करता है और टेक्स्ट युक्त .content विशेषता वाला एक ऑब्जेक्ट लौटाता है। इस कॉलेबल को एक `stop_sequences` आर्गुमेंट भी स्वीकार करने की आवश्यकता होती है जो बताता है कि कब जनरेट करना और बंद करना है।

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

model_id = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(model=model_id)

def custom_model(messages, stop_sequences=["Task"]):
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message
    return answer
```

इसके अतिरिक्त, `custom_model` एक `grammar` आर्गुमेंट भी ले सकता है। जिस स्थिति में आप एजेंट इनिशियलाइजेशन पर एक `grammar` निर्दिष्ट करते हैं, यह आर्गुमेंट मॉडल के कॉल्स को आपके द्वारा इनिशियलाइजेशन पर परिभाषित `grammar` के साथ पास किया जाएगा, ताकि [constrained generation](https://huggingface.co/docs/text-generation-inference/conceptual/guidance) की अनुमति मिल सके जिससे उचित-फॉर्मेटेड एजेंट आउटपुट को फोर्स किया जा सके।

### TransformersModel

सुविधा के लिए, हमने एक `TransformersModel` जोड़ा है जो इनिशियलाइजेशन पर दिए गए model_id के लिए एक लोकल `transformers` पाइपलाइन बनाकर ऊपर के बिंदुओं को लागू करता है।

```python
from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": "Ok!"}], stop_sequences=["great"]))
```
```text
>>> What a
```

[[autodoc]] TransformersModel

### InferenceClientModel

`InferenceClientModel` LLM के एक्जीक्यूशन के लिए [HF Inference API](https://huggingface.co/docs/api-inference/index) क्लाइंट को रैप करता है।

```python
from smolagents import InferenceClientModel

messages = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "No need to help, take it easy."},
]

model = InferenceClientModel()
print(model(messages))
```
```text
>>> Of course! If you change your mind, feel free to reach out. Take care!
```
[[autodoc]] InferenceClientModel

### LiteLLMModel

`LiteLLMModel` विभिन्न प्रदाताओं से 100+ LLMs को सपोर्ट करने के लिए [LiteLLM](https://www.litellm.ai/) का लाभ उठाता है।
आप मॉडल इनिशियलाइजेशन पर kwargs पास कर सकते हैं जो तब मॉडल का उपयोग करते समय प्रयोग किए जाएंगे, उदाहरण के लिए नीचे हम `temperature` पास करते हैं।

```python
from smolagents import LiteLLMModel

messages = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "No need to help, take it easy."},
]

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=10)
print(model(messages))
```

[[autodoc]] LiteLLMModel

### OpenAiServerModel


यह क्लास आपको किसी भी OpenAIServer कम्पैटिबल मॉडल को कॉल करने देती है।
यहाँ बताया गया है कि आप इसे कैसे सेट कर सकते हैं (आप दूसरे सर्वर को पॉइंट करने के लिए `api_base` url को कस्टमाइज़ कर सकते हैं):
```py
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

## Prompts

[[autodoc]] smolagents.agents.PromptTemplates

[[autodoc]] smolagents.agents.PlanningPromptTemplate

[[autodoc]] smolagents.agents.ManagedAgentPromptTemplate

[[autodoc]] smolagents.agents.FinalAnswerPromptTemplate
