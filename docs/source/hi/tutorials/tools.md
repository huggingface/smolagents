# Tools

[[open-in-colab]]

यहाँ, हम एडवांस्ड tools उपयोग देखेंगे।

> [!TIP]
> यदि आप एजेंट्स बनाने में नए हैं, तो सबसे पहले [एजेंट्स का परिचय](../conceptual_guides/intro_agents) और [smolagents की गाइडेड टूर](../guided_tour) पढ़ना सुनिश्चित करें।

- [Tools](#tools)
    - [टूल क्या है, और इसे कैसे बनाएं?](#टूल-क्या-है-और-इसे-कैसे-बनाएं)
    - [अपना टूल हब पर शेयर करें](#अपना-टूल-हब-पर-शेयर-करें)
    - [स्पेस को टूल के रूप में इम्पोर्ट करें](#स्पेस-को-टूल-के-रूप-में-इम्पोर्ट-करें)
    - [LangChain टूल्स का उपयोग करें](#LangChain-टूल्स-का-उपयोग-करें)
    - [अपने एजेंट के टूलबॉक्स को मैनेज करें](#अपने-एजेंट-के-टूलबॉक्स-को-मैनेज-करें)
    - [टूल्स का कलेक्शन उपयोग करें](#टूल्स-का-कलेक्शन-उपयोग-करें)

### टूल क्या है और इसे कैसे बनाएं

टूल मुख्य रूप से एक फ़ंक्शन है जिसे एक LLM एजेंटिक सिस्टम में उपयोग कर सकता है।

लेकिन इसका उपयोग करने के लिए, LLM को एक API दी जाएगी: नाम, टूल विवरण, इनपुट प्रकार और विवरण, आउटपुट प्रकार।

इसलिए यह केवल एक फ़ंक्शन नहीं हो सकता। यह एक क्लास होनी चाहिए।

तो मूल रूप से, टूल एक क्लास है जो एक फ़ंक्शन को मेटाडेटा के साथ रैप करती है जो LLM को समझने में मदद करती है कि इसका उपयोग कैसे करें।

यह कैसा दिखता है:

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()
```

कस्टम टूल `Tool` को सबक्लास करता है उपयोगी मेथड्स को इनहेरिट करने के लिए। चाइल्ड क्लास भी परिभाषित करती है:
- एक `name` एट्रिब्यूट, जो टूल के नाम से संबंधित है। नाम आमतौर पर बताता है कि टूल क्या करता है। चूंकि कोड एक टास्क के लिए सबसे अधिक डाउनलोड वाले मॉडल को रिटर्न करता है, इसलिए इसे `model_download_counter` नाम दें।
- एक `description` एट्रिब्यूट एजेंट के सिस्टम प्रॉम्प्ट को पॉपुलेट करने के लिए उपयोग किया जाता है।
- एक `inputs` एट्रिब्यूट, जो `"type"` और `"description"` keys वाला डिक्शनरी है। इसमें जानकारी होती है जो पायथन इंटरप्रेटर को इनपुट के बारे में शिक्षित विकल्प चुनने में मदद करती है।
- एक `output_type` एट्रिब्यूट, जो आउटपुट टाइप को निर्दिष्ट करता है। `inputs` और `output_type` दोनों के लिए टाइप [Pydantic formats](https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema) होने चाहिए, वे इनमें से कोई भी हो सकते हैं: `["string", "boolean","integer", "number", "image", "audio", "array", "object", "any", "null"]`।
- एक `forward` मेथड जिसमें एक्जीक्यूट किया जाने वाला इन्फरेंस कोड होता है।

एजेंट में उपयोग किए जाने के लिए इतना ही चाहिए!

टूल बनाने का एक और तरीका है। [guided_tour](../guided_tour) में, हमने `@tool` डेकोरेटर का उपयोग करके एक टूल को लागू किया। [`tool`] डेकोरेटर सरल टूल्स को परिभाषित करने का अनुशंसित तरीका है, लेकिन कभी-कभी आपको इससे अधिक की आवश्यकता होती है: अधिक स्पष्टता के लिए एक क्लास में कई मेथड्स का उपयोग करना, या अतिरिक्त क्लास एट्रिब्यूट्स का उपयोग करना।

इस स्थिति में, आप ऊपर बताए अनुसार [`Tool`] को सबक्लास करके अपना टूल बना सकते हैं।

### अपना टूल हब पर शेयर करें

आप टूल पर [`~Tool.push_to_hub`] को कॉल करके अपना कस्टम टूल हब पर शेयर कर सकते हैं। सुनिश्चित करें कि आपने हब पर इसके लिए एक रिपॉजिटरी बनाई है और आप रीड एक्सेस वाला टोकन उपयोग कर रहे हैं।

```python
model_downloads_tool.push_to_hub("{your_username}/hf-model-downloads", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

हब पर पुश करने के लिए काम करने के लिए, आपके टूल को कुछ नियमों का पालन करना होगा:
- सभी मेथड्स सेल्फ-कंटेन्ड हैं, यानी उनके आर्ग्स से आने वाले वेरिएबल्स का उपयोग करें।
- उपरोक्त बिंदु के अनुसार, **सभी इम्पोर्ट्स को सीधे टूल के फ़ंक्शंस के भीतर परिभाषित किया जाना चाहिए**, अन्यथा आपको अपने कस्टम टूल के साथ [`~Tool.save`] या [`~Tool.push_to_hub`] को कॉल करने का प्रयास करते समय एरर मिलेगा।
- यदि आप `__init__` विधि को सबक्लास करते हैं, तो आप इसे `self` के अलावा कोई अन्य आर्ग्यूमेंट नहीं दे सकते। ऐसा इसलिए है क्योंकि किसी विशिष्ट टूल इंस्टेंस के इनिशियलाइजेशन के दौरान सेट किए गए तर्कों को आर्ग्यूमेंट्स करना कठिन होता है, जो उन्हें हब पर ठीक से साझा करने से रोकता है। और वैसे भी, एक विशिष्ट क्लास बनाने का विचार यह है कि आप हार्ड-कोड के लिए आवश्यक किसी भी चीज़ के लिए क्लास विशेषताएँ पहले से ही सेट कर सकते हैं (बस `your_variable=(...)` को सीधे `class YourTool(Tool):` पंक्ति के अंतर्गत सेट करें ). और निश्चित रूप से आप अभी भी `self.your_variable` को असाइन करके अपने कोड में कहीं भी एक क्लास विशेषता बना सकते हैं।


एक बार जब आपका टूल हब पर पुश हो जाता है, तो आप इसे Space repository के रूप में देख सकते हैं। [यहाँ](https://huggingface.co/spaces/m-ric/hf-model-downloads/tree/main) `model_downloads_tool` है जिसे मैंने पुश किया है।

टूल फ़ाइलों में गहराई से जाने पर, आप पा सकते हैं कि सारी टूल लॉजिक [tool.py](https://huggingface.co/spaces/m-ric/hf-model-downloads/blob/main/tool.py) के अंतर्गत है। यहीं आप किसी और द्वारा शेयर किए गए टूल का निरीक्षण कर सकते हैं।

फिर आप टूल को [`load_tool`] के साथ लोड कर सकते हैं या [`~Tool.from_hub`] के साथ बना सकते हैं और इसे अपने एजेंट में `tools` पैरामीटर में पास कर सकते हैं।
चूंकि टूल्स को चलाने का मतलब कस्टम कोड चलाना है, आपको यह सुनिश्चित करना होगा कि आप रिपॉजिटरी पर भरोसा करते हैं, इसलिए हम हब से टूल लोड करने के लिए `trust_remote_code=True` पास करने की आवश्यकता रखते हैं।

```python
from smolagents import load_tool, CodeAgent

model_download_tool = load_tool(
    "{your_username}/hf-model-downloads",
    trust_remote_code=True
)
```

### स्पेस को टूल के रूप में इम्पोर्ट करें

आप [`Tool.from_space`] मेथड का उपयोग करके हब से एक स्पेस को सीधे टूल के रूप में इम्पोर्ट कर सकते हैं!

आपको केवल हब पर स्पेस की ID, इसका नाम, और एक विवरण प्रदान करने की आवश्यकता है जो आपके एजेंट को समझने में मदद करेगा कि टूल क्या करता है। अंदर से, यह स्पेस को कॉल करने के लिए [`gradio-client`](https://pypi.org/project/gradio-client/) लाइब्रेरी का उपयोग करेगा।

उदाहरण के लिए, चलिए हब से [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) स्पेस को इम्पोर्ट करें और इसका उपयोग एक इमेज जनरेट करने के लिए करें।

```python
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

image_generation_tool("A sunny beach")
```
और देखो, यह तुम्हारी छवि है! 🏖️

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sunny_beach.webp">

फिर आप इस टूल का उपयोग किसी अन्य टूल की तरह कर सकते हैं। उदाहरण के लिए, चलिए प्रॉम्प्ट `a rabbit wearing a space suit` को सुधारें और इसकी एक इमेज जनरेट करें। यह उदाहरण यह भी दिखाता है कि आप एजेंट को अतिरिक्त आर्ग्यूमेंट्स कैसे पास कर सकते हैं।

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")
agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.", additional_args={'user_prompt': 'A rabbit wearing a space suit'}
)
```

```text
=== Agent thoughts:
improved_prompt could be "A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background"

Now that I have improved the prompt, I can use the image generator tool to generate an image based on this prompt.
>>> Agent is executing the code below:
image = image_generator(prompt="A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background")
final_answer(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit_spacesuit_flux.webp">

यह कितना कूल है? 🤩

### LangChain टूल्स का उपयोग करें

हम LangChain को पसंद करते हैं और मानते हैं कि इसके पास टूल्स का एक बहुत आकर्षक संग्रह है।
LangChain से एक टूल इम्पोर्ट करने के लिए, `from_langchain()` मेथड का उपयोग करें।

यहाँ बताया गया है कि आप LangChain वेब सर्च टूल का उपयोग करके परिचय के सर्च रिजल्ट को कैसे फिर से बना सकते हैं।
इस टूल को काम करने के लिए `pip install langchain google-search-results -q` की आवश्यकता होगी।
```python
from langchain.agents import load_tools

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("How many more blocks (also denoted as layers) are in BERT base encoder compared to the encoder from the architecture proposed in Attention is All You Need?")
```

### अपने एजेंट के टूलबॉक्स को मैनेज करें

आप एजेंट के टूलबॉक्स को `agent.tools` एट्रिब्यूट में एक टूल जोड़कर या बदलकर मैनेज कर सकते हैं, क्योंकि यह एक स्टैंडर्ड डिक्शनरी है।

चलिए केवल डिफ़ॉल्ट टूलबॉक्स के साथ इनिशियलाइज़ किए गए मौजूदा एजेंट में `model_download_tool` जोड़ें।

```python
from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.tools[model_download_tool.name] = model_download_tool
```
अब हम नए टूल का लाभ उठा सकते हैं।

```python
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub but reverse the letters?"
)
```


> [!TIP]
> एजेंट में बहुत अधिक टूल्स न जोड़ने से सावधान रहें: यह कमजोर LLM इंजन को ओवरव्हेल्म कर सकता है।


### टूल्स का कलेक्शन उपयोग करें

आप `ToolCollection` ऑब्जेक्ट का उपयोग करके टूल कलेक्शंस का लाभ उठा सकते हैं। यह या तो हब से एक कलेक्शन या MCP सर्वर टूल्स को लोड करने का समर्थन करता है।

#### हब में कलेक्शन से टूल कलेक्शन

आप उस कलेक्शन के स्लग के साथ इसका लाभ उठा सकते हैं जिसका आप उपयोग करना चाहते हैं।
फिर उन्हें अपने एजेंट को इनिशियलाइज़ करने के लिए एक लिस्ट के रूप में पास करें, और उनका उपयोग शुरू करें!

```py
from smolagents import ToolCollection, CodeAgent

image_tool_collection = ToolCollection.from_hub(
    collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f",
    token="<YOUR_HUGGINGFACEHUB_API_TOKEN>"
)
agent = CodeAgent(tools=[*image_tool_collection.tools], model=model, add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes.")
```

स्टार्ट को तेज करने के लिए, टूल्स केवल तभी लोड होते हैं जब एजेंट द्वारा कॉल किए जाते हैं।

#### किसी भी MCP सर्वर से टूल कलेक्शन

[glama.ai](https://glama.ai/mcp/servers) या [smithery.ai](https://smithery.ai/) पर उपलब्ध सैकड़ों MCP सर्वर्स से टूल्स का लाभ उठाएं।

MCP सर्वर्स टूल्स को निम्नानुसार `ToolCollection` ऑब्जेक्ट में लोड किया जा सकता है:

```py
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

server_parameters = StdioServerParameters(
    command="uv",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True)
    agent.run("Please find a remedy for hangover.")
```
