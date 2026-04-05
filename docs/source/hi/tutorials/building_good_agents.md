# अच्छे Agents का निर्माण

[[open-in-colab]]

एक ऐसा एजेंट बनाने में जो काम करता है और जो काम नहीं करता है, इसमें ज़मीन-आसमान का अंतर है।
हम कैसे ऐसे एजेंट्स बना सकते हैं जो बाद वाली श्रेणी में आते हैं?
इस गाइड में, हम एजेंट्स बनाने के लिए सर्वोत्तम प्रक्रियाएँ के बारे में बात करेंगे।

> [!TIP]
> यदि आप एजेंट्स बनाने में नए हैं, तो पहले [एजेंट्स का परिचय](../conceptual_guides/intro_agents) और [smolagents की गाइडेड टूर](../guided_tour) पढ़ना सुनिश्चित करें।

### सर्वश्रेष्ठ एजेंटिक सिस्टम सबसे सरल होते हैं: वर्कफ़्लो को जितना हो सके उतना सरल बनाएं

अपने वर्कफ़्लो में एक LLM को कुछ एजेंसी देने से त्रुटियों का जोखिम होता है।

अच्छी तरह से प्रोग्राम किए गए एजेंटिक सिस्टम में वैसे भी अच्छी एरर लॉगिंग और रीट्राई मैकेनिज्म होते हैं, जिससे LLM इंजन अपनी गलतियों को सुधारने का मौका मिलता है। लेकिन LLM त्रुटि के जोखिम को अधिकतम कम करने के लिए, आपको अपना वर्कफ़्लो सरल बनाना चाहिए!

आइए [एजेंट्स का परिचय](../conceptual_guides/intro_agents) से उदाहरण पर फिर से विचार करें: एक सर्फ ट्रिप कंपनी के लिए उपयोगकर्ता प्रश्नों का उत्तर देने वाला बॉट।
एजेंट को हर बार जब एक नए सर्फ स्पॉट के बारे में पूछा जाता है तो "travel distance API" और "weather API" के लिए 2 अलग-अलग कॉल करने देने के बजाय, आप केवल एक एकीकृत टूल "return_spot_information" बना सकते हैं, एक फंक्शन जो दोनों APIs को एक साथ कॉल करता है और उनके संयोजित आउटपुट को उपयोगकर्ता को वापस करता है।

यह लागत, देरी और त्रुटि जोखिम को कम करेगा!

मुख्य दिशानिर्देश है: LLM कॉल्स की संख्या को जितना हो सके उतना कम करें।

इससे कुछ निष्कर्ष निकलते हैं:
- जब भी संभव हो, दो APIs के हमारे उदाहरण की तरह 2 टूल्स को एक में समूहित करें।
- जब भी संभव हो, लॉजिक एजेंटिक निर्णयों के बजाय डिटरमिनिस्टिक फंक्शंस पर आधारित होनी चाहिए।

### LLM इंजन को जानकारी के प्रवाह में सुधार करें

याद रखें कि आपका LLM इंजन एक *बुद्धिमान* रोबोट की तरह है, जो एक कमरे में बंद है, और बाहरी दुनिया के साथ इसका एकमात्र संचार दरवाजे के नीचे से नोट्स पास करना है।

यह किसी भी ऐसी चीज के बारे में नहीं जानेगा जिसे आप स्पष्ट रूप से अपने प्रॉम्प्ट में नहीं डालते हैं।

इसलिए पहले अपने कार्य को बहुत स्पष्ट बनाने से शुरू करें!
चूंकि एक एजेंट LLM द्वारा संचालित होता है, आपके कार्य के निर्माण में छोटे बदलाव भी पूरी तरह से अलग परिणाम दे सकते हैं।

फिर, टूल के उपयोग में अपने एजेंट की ओर जानकारी के प्रवाह में सुधार करें।

पालन करने के लिए विशेष दिशानिर्देश:
- प्रत्येक टूल को वह सब कुछ लॉग करना चाहिए (टूल की `forward` मेथड के अंदर केवल `print` स्टेटमेंट्स का उपयोग करके) जो LLM इंजन के लिए उपयोगी हो सकता है।
  - विशेष रूप से, टूल एक्जीक्यूशन गलतियों पर विस्तृत लॉगिंग बहुत मदद करेगी!

उदाहरण के लिए, यहाँ एक टूल है जो लोकेशन और डेट-टाइम के आधार पर मौसम डेटा प्राप्त करता है:

पहले, यहाँ एक खराब रूप है:
```python
import datetime
from smolagents import tool

def get_weather_report_at_coordinates(coordinates, date_time):
    # Dummy function, returns a list of [temperature in °C, risk of rain on a scale 0-1, wave height in m]
    return [28.0, 0.35, 0.85]

def convert_location_to_coordinates(location):
    # Returns dummy coordinates
    return [3.3, -42.0]

@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for.
        date_time: the date and time for which you want the report.
    """
    lon, lat = convert_location_to_coordinates(location)
    date_time = datetime.strptime(date_time)
    return str(get_weather_report_at_coordinates((lon, lat), date_time))
```

# यह खराब क्यों है?
- `date_time` के लिए उपयोग किए जाने वाले फॉर्मेट की सटीकता का कोई उल्लेख नहीं है।  
- यह स्पष्ट नहीं है कि स्थान (location) को किस प्रकार निर्दिष्ट किया जाना चाहिए।  
- त्रुटियों को स्पष्ट रूप से इंगित करने के लिए कोई लॉगिंग मेकैनिज्म मौजूद नहीं है, जैसे कि स्थान गलत फॉर्मेट में होना या `date_time` का सही ढंग से फॉर्मेट न होना।  
- आउटपुट फॉर्मेट समझने में कठिन है।  

यदि टूल कॉल विफल हो जाती है, तो मेमोरी में लॉग की गई एरर ट्रेस LLM को टूल की समस्याओं को ठीक करने के लिए रिवर्स इंजीनियरिंग में मदद कर सकती है। लेकिन इतना सारा काम LLM को ही क्यों करने देना?

इस टूल को बेहतर तरीके से बनाने का एक उदाहरण इस प्रकार हो सकता है:

```python
@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".
        date_time: the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    lon, lat = convert_location_to_coordinates(location)
    try:
        date_time = datetime.strptime(date_time)
    except Exception as e:
        raise ValueError("Conversion of `date_time` to datetime format failed, make sure to provide a string in format '%m/%d/%y %H:%M:%S'. Full trace:" + str(e))
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates((lon, lat), date_time)
    return f"Weather report for {location}, {date_time}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain*100:.0f}%, wave height is {wave_height}m."
```

सामान्य तौर पर, अपने LLM का बोझ को कम करने के लिए, खुद से यह अच्छा सवाल पूछें: "यदि मैं नया और अनुभवहीन हूं और इस टूल का पहली बार उपयोग कर रहा हूं, तो इस टूल के साथ प्रोग्रामिंग करना और अपनी गलतियों को ठीक करना मेरे लिए कितना आसान होगा?"

### एजेंट को अधिक तर्क (arguments) दें

अपने एजेंट को कार्य का वर्णन करने वाले साधारण स्ट्रिंग से आगे बढ़कर कुछ अतिरिक्त ऑब्जेक्ट्स देने के लिए, आप `additional_args` का उपयोग कर सकते हैं। यह आपको किसी भी प्रकार का ऑब्जेक्ट पास करने की सुविधा देता है:


```py
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

agent = CodeAgent(tools=[], model=InferenceClientModel(model_id=model_id), add_base_tools=True)

agent.run(
    "Why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
)
```
उदाहरण के लिए, आप इस `additional_args` आर्ग्यूमेंट का उपयोग उन इमेजेज़ या स्ट्रिंग्स को पास करने के लिए कर सकते हैं जिन्हें आप चाहते हैं कि आपका एजेंट उपयोग करे।



## अपने एजेंट को डिबग कैसे करें

### 1. एक अधिक शक्तिशाली LLM का उपयोग करें

एजेंटिक वर्कफ़्लो में, कुछ त्रुटियां वास्तविक होती हैं, जबकि कुछ अन्य त्रुटियां आपके LLM इंजन के सही तरीके से तर्क न कर पाने की वजह से होती हैं।  
उदाहरण के लिए, इस ट्रेस को देखें, जहां मैंने एक `CodeAgent` से एक कार की तस्वीर बनाने के लिए कहा:
```
==================================================================================================== New task ====================================================================================================
Make me a cool car picture
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
image_generator(prompt="A cool, futuristic sports car with LED headlights, aerodynamic design, and vibrant color, high-res, photorealistic")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Step 1:

- Time taken: 16.35 seconds
- Input tokens: 1,383
- Output tokens: 77
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
final_answer("/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Print outputs:

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Final answer:
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
```
उपयोगकर्ता को, एक इमेज लौटाए जाने के बजाय, उन्हें एक पाथ लौटाया जाता है।
यह सिस्टम से एक बग की तरह दिख सकता है, लेकिन वास्तव में एजेंटिक सिस्टम ने त्रुटि नहीं की: यह केवल इसलिए है कि LLM ब्रेन ने इमेज आउटपुट को एक वेरिएबल में सेव करने की गलती की।
इस प्रकार यह इमेज को फिर से एक्सेस नहीं कर सकता है सिवाय इमेज को सेव करते समय लॉग किए गए पाथ का उपयोग करके, इसलिए यह इमेज के बजाय पाथ लौटाता है।

अपने एजेंट को डीबग करने का पहला कदम इस प्रकार है "एक अधिक शक्तिशाली LLM का उपयोग करें"। `Qwen2/5-72B-Instruct` जैसे विकल्प वह गलती नहीं करते।

### 2. अधिक मार्गदर्शन / अधिक जानकारी प्रदान करें

आप कम शक्तिशाली मॉडल्स का भी उपयोग कर सकते हैं, बशर्ते आप उन्हें अधिक प्रभावी ढंग से मार्गदर्शन करें।

अपने आप को अपने मॉडल की जगह रखें: यदि आप कार्य को हल करने वाला मॉडल होते, तो क्या आप उपलब्ध जानकारी (सिस्टम प्रॉम्प्ट + कार्य निर्माण + टूल विवरण से) के साथ संघर्ष करते?

क्या आपको कुछ अतिरिक्त स्पष्टीकरण की आवश्यकता होती?

अतिरिक्त जानकारी प्रदान करने के लिए, हम तुरंत सिस्टम प्रॉम्प्ट को बदलने की सलाह नहीं देते हैं: डिफ़ॉल्ट सिस्टम प्रॉम्प्ट में कई समायोजन हैं जिन्हें आप तब तक नहीं बिगाड़ना चाहते जब तक आप प्रॉम्प्ट को बहुत अच्छी तरह से नहीं समझते।
अपने LLM इंजन को मार्गदर्शन करने के बेहतर तरीके हैं:
- यदि यह कार्य को हल करने के बारे में है: इन सभी विवरणों को कार्य में जोड़ें। यह कार्य 100 पेज लंबा हो सकता है
- यदि यह टूल्स के उपयोग के बारे में है: आपके टूल्स की विवरण विशेषता।

### 3. सिस्टम प्रॉम्प्ट बदलें (आमतौर पर यह सलाह नहीं दी जाती)

यदि उपरोक्त स्पष्टीकरण पर्याप्त नहीं हैं, तो आप सिस्टम प्रॉम्प्ट बदल सकते हैं।

आइए देखें कि यह कैसे काम करता है। उदाहरण के लिए, आइए [`CodeAgent`] के लिए डिफ़ॉल्ट सिस्टम प्रॉम्प्ट की जाँच करें (नीचे दिया गया वर्जन जीरो-शॉट उदाहरणों को छोड़कर छोटा किया गया है)।

```python
print(agent.prompt_templates["system_prompt"])
```
Here is what you get:
```text
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```<end_code>
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
Code:
```py
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
```<end_code>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
Code:
```py
result = 5 + 3 + 1294.678
final_answer(result)
```<end_code>

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
```<end_code>

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
```<end_code>
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam")
print(pages)
```<end_code>
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
Code:
```py
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("\n" + "="*80 + "\n")  # Print separator between pages
```<end_code>
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
Code:
```py
final_answer("diminished")
```<end_code>

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Code:
```py
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {city}:", search(f"{city} population")
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `wiki` to get the age of the pope, and confirm that with a web search.
Code:
```py
pope_age_wiki = wiki(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
```<end_code>
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
```<end_code>

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:
{%- for tool in tools.values() %}
- {{ tool.to_tool_calling_prompt() }}
{%- endfor %}

{%- if managed_agents and managed_agents.values() | list %}
You can also give tasks to team members.
Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
You can also include any relevant variables or context using the 'additional_args' argument.
Here is a list of the team members that you can call:
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
{%- endif %}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
```

जैसा कि आप देख सकते हैं, `"{{ tool.description }}"` जैसे प्लेसहोल्डर्स हैं: इनका उपयोग एजेंट इनिशियलाइजेशन के समय टूल्स या मैनेज्ड एजेंट्स के कुछ स्वचालित रूप से जनरेट किए गए विवरणों को डालने के लिए किया जाएगा।

इसलिए जबकि आप `system_prompt` पैरामीटर में अपने कस्टम प्रॉम्प्ट को आर्गुमेंट के रूप में पास करके इस सिस्टम प्रॉम्प्ट टेम्पलेट को ओवरराइट कर सकते हैं, आपके नए सिस्टम प्रॉम्प्ट में निम्नलिखित प्लेसहोल्डर्स होने चाहिए:
- टूल विवरण डालने के लिए।
  ```
  {%- for tool in tools.values() %}
  - {{ tool.to_tool_calling_prompt() }}
  {%- endfor %}
  ```
- यदि कोई मैनेज्ड एजेंट्स हैं तो उनके लिए विवरण डालने के लिए।
  ```
  {%- if managed_agents and managed_agents.values() | list %}
  You can also give tasks to team members.
  Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
  You can also include any relevant variables or context using the 'additional_args' argument.
  Here is a list of the team members that you can call:
  {%- for agent in managed_agents.values() %}
  - {{ agent.name }}: {{ agent.description }}
  {%- endfor %}
  {%- endif %}
  ```
- केवल `CodeAgent` के लिए: अधिकृत इम्पोर्ट्स की सूची डालने के लिए `"{{authorized_imports}}"`।

फिर आप सिस्टम प्रॉम्प्ट को निम्नानुसार बदल सकते हैं:

```py
agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + "\nHere you go!"
```

This also works with the [`ToolCallingAgent`].


### 4. अतिरिक्त योजना

हम पूरक योजना चरण के लिए एक मॉडल प्रदान करते हैं, जिसे एजेंट सामान्य क्रियाओं के चरणों के बीच नियमित रूप से चला सकता है। इस चरण में कोई टूल कॉल नहीं होती है, LLM से केवल उन तथ्यों की सूची को अपडेट करने के लिए कहा जाता है जो उसे ज्ञात हैं और इन तथ्यों के आधार पर उसे अगले कदमों के बारे में विचार करना होता है।

```py
from smolagents import load_tool, CodeAgent, InferenceClientModel, WebSearchTool
from dotenv import load_dotenv

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = WebSearchTool()

agent = CodeAgent(
    tools=[search_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct"),
    planning_interval=3 # This is where you activate planning!
)

# Run it!
result = agent.run(
    "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
)
```
