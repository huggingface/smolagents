# Agents - गाइडेड टूर

[[open-in-colab]]

इस गाइडेड विजिट में, आप सीखेंगे कि एक एजेंट कैसे बनाएं, इसे कैसे चलाएं, और अपने यूज-केस के लिए बेहतर काम करने के लिए इसे कैसे कस्टमाइज़ करें।

### अपना Agent बनाना

एक मिनिमल एजेंट को इनिशियलाइज़ करने के लिए, आपको कम से कम इन दो आर्ग्यूमेंट्स की आवश्यकता है:

- `model`, आपके एजेंट को पावर देने के लिए एक टेक्स्ट-जनरेशन मॉडल - क्योंकि एजेंट एक सिंपल LLM से अलग है, यह एक सिस्टम है जो LLM को अपने इंजन के रूप में उपयोग करता है। आप इनमें से कोई भी विकल्प उपयोग कर सकते हैं:
    - [`TransformersModel`] `transformers` पाइपलाइन को पहले से इनिशियलाइज़ करता है जो `transformers` का उपयोग करके आपकी लोकल मशीन पर इन्फरेंस चलाने के लिए होता है।
    - [`InferenceClientModel`] अंदर से `huggingface_hub.InferenceClient` का लाभ उठाता है।
    - [`LiteLLMModel`] आपको [LiteLLM](https://docs.litellm.ai/) के माध्यम से 100+ अलग-अलग मॉडल्स को कॉल करने देता है!

- `tools`, `Tools` की एक लिस्ट जिसे एजेंट टास्क को हल करने के लिए उपयोग कर सकता है। यह एक खाली लिस्ट हो सकती है। आप ऑप्शनल आर्ग्यूमेंट `add_base_tools=True` को परिभाषित करके अपनी `tools` लिस्ट के ऊपर डिफ़ॉल्ट टूलबॉक्स भी जोड़ सकते हैं।

एक बार जब आपके पास ये दो आर्ग्यूमेंट्स, `tools` और `model` हैं, तो आप एक एजेंट बना सकते हैं और इसे चला सकते हैं। आप कोई भी LLM उपयोग कर सकते हैं, या तो [Hugging Face API](https://huggingface.co/docs/api-inference/en/index), [transformers](https://github.com/huggingface/transformers/), [ollama](https://ollama.com/), या [LiteLLM](https://www.litellm.ai/) के माध्यम से।

<hfoptions id="एक LLM चुनें">
<hfoption id="Hugging Face API">

Hugging Face API टोकन के बिना उपयोग करने के लिए मुफ्त है, लेकिन फिर इसमें रेट लिमिटेशन होगी।

गेटेड मॉडल्स तक पहुंचने या PRO अकाउंट के साथ अपनी रेट लिमिट्स बढ़ाने के लिए, आपको एनवायरनमेंट वेरिएबल `HF_TOKEN` सेट करना होगा या `InferenceClientModel` के इनिशियलाइजेशन पर `token` वेरिएबल पास करना होगा।

```python
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = InferenceClientModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Local Transformers Model">

```python
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="OpenAI या Anthropic API">

`LiteLLMModel` का उपयोग करने के लिए, आपको एनवायरनमेंट वेरिएबल `ANTHROPIC_API_KEY` या `OPENAI_API_KEY` सेट करना होगा, या इनिशियलाइजेशन पर `api_key` वेरिएबल पास करना होगा।

```python
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY") # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Ollama">

```python
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2", # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_key="YOUR_API_KEY" # replace with API key if necessary
    num_ctx=8192 # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
</hfoptions>

#### CodeAgent और ToolCallingAgent

[`CodeAgent`] हमारा डिफ़ॉल्ट एजेंट है। यह हर स्टेप पर पायथन कोड स्निपेट्स लिखेगा और एक्जीक्यूट करेगा।

डिफ़ॉल्ट रूप से, एक्जीक्यूशन आपके लोकल एनवायरनमेंट में किया जाता है।
यह सुरक्षित होना चाहिए क्योंकि केवल वही फ़ंक्शंस कॉल किए जा सकते हैं जो आपने प्रदान किए हैं (विशेष रूप से यदि यह केवल Hugging Face टूल्स हैं) और पूर्व-परिभाषित सुरक्षित फ़ंक्शंस जैसे `print` या `math` मॉड्यूल से फ़ंक्शंस, इसलिए आप पहले से ही सीमित हैं कि क्या एक्जीक्यूट किया जा सकता है।

पायथन इंटरप्रेटर डिफ़ॉल्ट रूप से सेफ लिस्ट के बाहर इम्पोर्ट की अनुमति नहीं देता है, इसलिए सबसे स्पष्ट अटैक समस्या नहीं होनी चाहिए।
आप अपने [`CodeAgent`] के इनिशियलाइजेशन पर आर्ग्यूमेंट `additional_authorized_imports` में स्ट्रिंग्स की लिस्ट के रूप में अतिरिक्त मॉड्यूल्स को अधिकृत कर सकते हैं।

```py
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

> [!WARNING]
> LLM आर्बिट्ररी कोड जनरेट कर सकता है जो फिर एक्जीक्यूट किया जाएगा: कोई असुरक्षित इम्पोर्ट न जोड़ें!

एक्जीक्यूशन किसी भी कोड पर रुक जाएगा जो एक अवैध ऑपरेशन करने का प्रयास करता है या यदि एजेंट द्वारा जनरेट किए गए कोड में एक रेगुलर पायथन एरर है।

आप [E2B कोड एक्जीक्यूटर](https://e2b.dev/docs#what-is-e2-b) या Docker का उपयोग लोकल पायथन इंटरप्रेटर के बजाय कर सकते हैं। E2B के लिए, पहले [`E2B_API_KEY` एनवायरनमेंट वेरिएबल सेट करें](https://e2b.dev/dashboard?tab=keys) और फिर एजेंट इनिशियलाइजेशन पर `executor_type="e2b"` पास करें। Docker के लिए, इनिशियलाइजेशन के दौरान `executor_type="docker"` पास करें।

> [!TIP]
> कोड एक्जीक्यूशन के बारे में और जानें [इस ट्यूटोरियल में](tutorials/secure_code_execution)।

हम JSON-जैसे ब्लॉब्स के रूप में एक्शन लिखने के व्यापक रूप से उपयोग किए जाने वाले तरीके का भी समर्थन करते हैं: यह [`ToolCallingAgent`] है, यह बहुत कुछ [`CodeAgent`] की तरह ही काम करता है, बेशक `additional_authorized_imports` के बिना क्योंकि यह कोड एक्जीक्यूट नहीं करता।

```py
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(tools=[], model=model)
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

### एजेंट रन का निरीक्षण

रन के बाद क्या हुआ यह जांचने के लिए यहाँ कुछ उपयोगी एट्रिब्यूट्स हैं:
- `agent.logs` एजेंट के फाइन-ग्रेन्ड लॉग्स को स्टोर करता है। एजेंट के रन के हर स्टेप पर, सब कुछ एक डिक्शनरी में स्टोर किया जाता है जो फिर `agent.logs` में जोड़ा जाता है।
- `agent.write_memory_to_messages()` चलाने से LLM के लिए एजेंट के लॉग्स की एक इनर मेमोरी बनती है, चैट मैसेज की लिस्ट के रूप में। यह मेथड लॉग के प्रत्येक स्टेप पर जाता है और केवल वही स्टोर करता है जिसमें यह एक मैसेज के रूप में रुचि रखता है: उदाहरण के लिए, यह सिस्टम प्रॉम्प्ट और टास्क को अलग-अलग मैसेज के रूप में सेव करेगा, फिर प्रत्येक स्टेप के लिए यह LLM आउटपुट को एक मैसेज के रूप में और टूल कॉल आउटपुट को दूसरे मैसेज के रूप में स्टोर करेगा।

## टूल्स

टूल एक एटॉमिक फ़ंक्शन है जिसे एजेंट द्वारा उपयोग किया जाता है। LLM द्वारा उपयोग किए जाने के लिए, इसे कुछ एट्रिब्यूट्स की भी आवश्यकता होती है जो इसकी API बनाते हैं और LLM को यह बताने के लिए उपयोग किए जाएंगे कि इस टूल को कैसे कॉल करें:
- एक नाम
- एक विवरण
- इनपुट प्रकार और विवरण
- एक आउटपुट प्रकार

आप उदाहरण के लिए [`PythonInterpreterTool`] को चेक कर सकते हैं: इसमें एक नाम, विवरण, इनपुट विवरण, एक आउटपुट प्रकार, और एक्शन करने के लिए एक `forward` मेथड है।

जब एजेंट इनिशियलाइज़ किया जाता है, टूल एट्रिब्यूट्स का उपयोग एक टूल विवरण जनरेट करने के लिए किया जाता है जो एजेंट के सिस्टम प्रॉम्प्ट में बेक किया जाता है। यह एजेंट को बताता है कि वह कौन से टूल्स उपयोग कर सकता है और क्यों।

### डिफ़ॉल्ट टूलबॉक्स

`smolagents` एजेंट्स को सशक्त बनाने के लिए एक डिफ़ॉल्ट टूलबॉक्स के साथ आता है, जिसे आप आर्ग्यूमेंट `add_base_tools=True` के साथ अपने एजेंट में इनिशियलाइजेशन पर जोड़ सकते हैं:

- **DuckDuckGo वेब सर्च**: DuckDuckGo ब्राउज़र का उपयोग करके वेब सर्च करता है।
- **पायथन कोड इंटरप्रेटर**: आपका LLM जनरेटेड पायथन कोड एक सुरक्षित एनवायरनमेंट में चलाता है। यह टूल [`ToolCallingAgent`] में केवल तभी जोड़ा जाएगा जब आप इसे `add_base_tools=True` के साथ इनिशियलाइज़ करते हैं, क्योंकि कोड-बेस्ड एजेंट पहले से ही नेटिव रूप से पायथन कोड एक्जीक्यूट कर सकता है
- **ट्रांसक्राइबर**: Whisper-Turbo पर बनाया गया एक स्पीच-टू-टेक्स्ट पाइपलाइन जो ऑडियो को टेक्स्ट में ट्रांसक्राइब करता है।

आप मैन्युअल रूप से एक टूल का उपयोग उसके आर्ग्यूमेंट्स के साथ कॉल करके कर सकते हैं।

```python
from smolagents import WebSearchTool

search_tool = WebSearchTool()
print(search_tool("Who's the current president of Russia?"))
```

### अपने कस्टम टूल बनाएं  

आप ऐसे उपयोग के मामलों के लिए अपने खुद के टूल बना सकते हैं जो Hugging Face के डिफ़ॉल्ट टूल्स द्वारा कवर नहीं किए गए हैं।  
उदाहरण के लिए, चलिए एक टूल बनाते हैं जो दिए गए कार्य (task) के लिए हब से सबसे अधिक डाउनलोड किए गए मॉडल को रिटर्न करता है।  

आप नीचे दिए गए कोड से शुरुआत करेंगे। 

```python
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(most_downloaded_model.id)
```

यह कोड आसानी से टूल में बदला जा सकता है, बस इसे एक फ़ंक्शन में रैप करें और `tool` डेकोरेटर जोड़ें:  
यह टूल बनाने का एकमात्र तरीका नहीं है: आप इसे सीधे [`Tool`] का सबक्लास बनाकर भी परिभाषित कर सकते हैं, जो आपको अधिक लचीलापन प्रदान करता है, जैसे भारी क्लास एट्रिब्यूट्स को इनिशियलाइज़ करने की संभावना।  

चलो देखते हैं कि यह दोनों विकल्पों के लिए कैसे काम करता है:

<hfoptions id="build-a-tool">
<hfoption id="@tool के साथ एक फ़ंक्शन को डेकोरेट करें">

```py
from smolagents import tool

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id
```

फ़ंक्शन को चाहिए:  
- एक स्पष्ट नाम: नाम टूल के कार्य को स्पष्ट रूप से बताने वाला होना चाहिए ताकि इसे चलाने वाले LLM को आसानी हो। चूंकि यह टूल कार्य के लिए सबसे अधिक डाउनलोड किए गए मॉडल को लौटाता है, इसका नाम `model_download_tool` रखा गया है।  
- इनपुट और आउटपुट पर टाइप हिंट्स।
- एक विवरण: इसमें 'Args:' भाग शामिल होना चाहिए, जिसमें प्रत्येक आर्ग्युमेंट का वर्णन (बिना टाइप संकेत के) किया गया हो। यह विवरण एक निर्देश मैनुअल की तरह होता है जो LLM को टूल चलाने में मदद करता है। इसे अनदेखा न करें।  
इन सभी तत्वों को एजेंट की सिस्टम प्रॉम्प्ट में स्वचालित रूप से शामिल किया जाएगा: इसलिए इन्हें यथासंभव स्पष्ट बनाने का प्रयास करें!  

> [!TIP]  
> यह परिभाषा प्रारूप `apply_chat_template` में उपयोग की गई टूल स्कीमा जैसा ही है, केवल अतिरिक्त `tool` डेकोरेटर जोड़ा गया है: हमारे टूल उपयोग API के बारे में अधिक पढ़ें [यहाँ](https://huggingface.co/blog/unified-tool-use#passing-tools-to-a-chat-template)।  
</hfoption>
<hfoption id="सबक्लास टूल">

```py
from smolagents import Tool

class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return most_downloaded_model.id
```

सबक्लास को निम्नलिखित एट्रिब्यूट्स की आवश्यकता होती है:  
- एक स्पष्ट `name`: नाम टूल के कार्य को स्पष्ट रूप से बताने वाला होना चाहिए।  
- एक `description`: यह भी LLM के लिए निर्देश मैनुअल की तरह काम करता है।  
- इनपुट प्रकार और उनके विवरण।  
- आउटपुट प्रकार।  
इन सभी एट्रिब्यूट्स को एजेंट की सिस्टम प्रॉम्प्ट में स्वचालित रूप से शामिल किया जाएगा, इन्हें स्पष्ट और विस्तृत बनाएं।  
</hfoption>
</hfoptions>


आप सीधे अपने एजेंट को इनिशियलाइज़ कर सकते हैं:  
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

लॉग्स इस प्रकार होंगे:  
```text
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                          │
│ Can you give me the name of the model that has the most downloads in the 'text-to-video' │
│ task on the Hugging Face Hub?                                                            │
│                                                                                          │
╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 model_name = model_download_tool(task="text-to-video")                               │
│   2 print(model_name)                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
ByteDance/AnimateDiff-Lightning

Out: None
[Step 0: Duration 0.27 seconds| Input tokens: 2,069 | Output tokens: 60]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer("ByteDance/AnimateDiff-Lightning")                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: ByteDance/AnimateDiff-Lightning
[Step 1: Duration 0.10 seconds| Input tokens: 4,288 | Output tokens: 148]
Out[20]: 'ByteDance/AnimateDiff-Lightning'
```

 [!TIP]  
> टूल्स के बारे में अधिक पढ़ें [dedicated tutorial](./tutorials/tools#टूल-क्या-है-और-इसे-कैसे-बनाएं) में।  

## मल्टी-एजेंट्स  

Microsoft के फ्रेमवर्क [Autogen](https://huggingface.co/papers/2308.08155) के साथ मल्टी-एजेंट सिस्टम्स की शुरुआत हुई।  

इस प्रकार के फ्रेमवर्क में, आपके कार्य को हल करने के लिए कई एजेंट्स एक साथ काम करते हैं, न कि केवल एक।  
यह अधिकांश बेंचमार्क्स पर बेहतर प्रदर्शन देता है। इसका कारण यह है कि कई कार्यों के लिए, एक सर्व-समावेशी प्रणाली के बजाय, आप उप-कार्यों पर विशेषज्ञता रखने वाली इकाइयों को पसंद करेंगे।  इस तरह, अलग-अलग टूल सेट्स और मेमोरी वाले एजेंट्स के पास विशेषकरण की अधिक कुशलता होती है। उदाहरण के लिए, कोड उत्पन्न करने वाले एजेंट की मेमोरी को वेब सर्च एजेंट द्वारा देखे गए वेबपेजों की सभी सामग्री से क्यों भरें? इन्हें अलग रखना बेहतर है।  

आप `smolagents` का उपयोग करके आसानी से श्रेणीबद्ध मल्टी-एजेंट सिस्टम्स बना सकते हैं।  

ऐसा करने के लिए, एजेंट को [`ManagedAgent`] ऑब्जेक्ट में समाहित करें। यह ऑब्जेक्ट `agent`, `name`, और एक `description` जैसे तर्कों की आवश्यकता होती है, जो फिर मैनेजर एजेंट की सिस्टम प्रॉम्प्ट में एम्बेड किया जाता है  

यहां एक एजेंट बनाने का उदाहरण दिया गया है जो हमारे [`WebSearchTool`] का उपयोग करके एक विशिष्ट वेब खोज एजेंट को प्रबंधित करता है।

```py
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, ManagedAgent

model = InferenceClientModel()

web_agent = CodeAgent(tools=[WebSearchTool()], model=model)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[managed_web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> कुशल मल्टी-एजेंट इंप्लीमेंटेशन का एक विस्तृत उदाहरण देखने के लिए, [कैसे हमने अपने मल्टी-एजेंट सिस्टम को GAIA लीडरबोर्ड के शीर्ष पर पहुंचाया](https://huggingface.co/blog/beating-gaia) पर जाएं।  


## अपने एजेंट से बात करें और उसके विचारों को एक शानदार Gradio इंटरफेस में विज़ुअलाइज़ करें  

आप `GradioUI` का उपयोग करके अपने एजेंट को इंटरैक्टिव तरीके से कार्य सौंप सकते हैं और उसके सोचने और निष्पादन की प्रक्रिया को देख सकते हैं। नीचे एक उदाहरण दिया गया है:

```py
from smolagents import (
    load_tool,
    CodeAgent,
    InferenceClientModel,
    GradioUI
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = InferenceClientModel(model_id=model_id)

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
```

अंदरूनी तौर पर, जब यूजर एक नया उत्तर टाइप करता है, तो एजेंट को `agent.run(user_request, reset=False)` के साथ लॉन्च किया जाता है।  
यहाँ `reset=False` फ्लैग का मतलब है कि एजेंट की मेमोरी इस नए कार्य को लॉन्च करने से पहले क्लियर नहीं होती, जिससे बातचीत जारी रहती है।  

आप इस `reset=False` आर्ग्युमेंट का उपयोग किसी भी अन्य एजेंटिक एप्लिकेशन में बातचीत जारी रखने के लिए कर सकते हैं।  

## अगले कदम  

अधिक गहन उपयोग के लिए, आप हमारे ट्यूटोरियल्स देख सकते हैं:  
- [हमारे कोड एजेंट्स कैसे काम करते हैं इसका विवरण](./tutorials/secure_code_execution)  
- [अच्छे एजेंट्स बनाने के लिए यह गाइड](./tutorials/building_good_agents)  
- [टूल उपयोग के लिए इन-डेप्थ गाइड ](./tutorials/building_good_agents)।  
