# मल्टी-एजेंट सिस्टम का आयोजन करें 🤖🤝🤖

[[open-in-colab]]

इस नोटबुक में हम एक **मल्टी-एजेंट वेब ब्राउज़र बनाएंगे: एक एजेंटिक सिस्टम जिसमें कई एजेंट वेब का उपयोग करके समस्याओं को हल करने के लिए सहयोग करते हैं!**

यह एक सरल संरचना होगी, जो प्रबंधित वेब खोज एजेंट को रैप करने के लिए `ManagedAgent` ऑब्जेक्ट का उपयोग करता है:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+
```
आइए इस सिस्टम को सेट करें।

आवश्यक डिपेंडेंसी इंस्टॉल करने के लिए नीचे दी गई लाइन चलाएं:

```
!pip install 'smolagents[toolkit]' --upgrade -q
```

HF Inference API को कॉल करने के लिए लॉगिन करें:

```
from huggingface_hub import login

login()
```

⚡️ हमारा एजेंट [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) द्वारा संचालित होगा जो `InferenceClientModel` क्लास का उपयोग करता है जो HF के Inference API का उपयोग करता है: Inference API किसी भी OS मॉडल को जल्दी और आसानी से चलाने की अनुमति देता है।

_नोट:_ The Inference API विभिन्न मानदंडों के आधार पर मॉडल होस्ट करता है, और डिप्लॉय किए गए मॉडल बिना पूर्व सूचना के अपडेट या बदले जा सकते हैं। इसके बारे में अधिक जानें [यहां](https://huggingface.co/docs/api-inference/supported-models)।

```py
model_id = "Qwen/Qwen3-4B-Instruct-2507"
```

## 🔍 एक वेब सर्च टूल बनाएं

वेब ब्राउज़िंग के लिए, हम पहले से मौजूद [`WebSearchTool`] टूल का उपयोग कर सकते हैं जो Google search के समान सुविधा प्रदान करता है।

लेकिन फिर हमें `WebSearchTool` द्वारा खोजे गए पेज को देखने में भी सक्षम होने की आवश्यकता होगी।
ऐसा करने के लिए, हम लाइब्रेरी के बिल्ट-इन `VisitWebpageTool` को इम्पोर्ट कर सकते हैं, लेकिन हम इसे फिर से बनाएंगे यह देखने के लिए कि यह कैसे किया जाता है।

तो आइए `markdownify` का उपयोग करके शुरू से अपना `VisitWebpageTool` टूल बनाएं।

```py
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

ठीक है, अब चलिए हमारे टूल को टेस्ट करें!

```py
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## हमारी मल्टी-एजेंट सिस्टम का निर्माण करें 🤖🤝🤖

अब जब हमारे पास सभी टूल्स `search` और `visit_webpage` हैं, हम उनका उपयोग वेब एजेंट बनाने के लिए कर सकते हैं।

इस एजेंट के लिए कौन सा कॉन्फ़िगरेशन चुनें?
- वेब ब्राउज़िंग एक सिंगल-टाइमलाइन टास्क है जिसे समानांतर टूल कॉल की आवश्यकता नहीं है, इसलिए JSON टूल कॉलिंग इसके लिए अच्छी तरह काम करती है। इसलिए हम `ToolCallingAgent` चुनते हैं।
- साथ ही, चूंकि कभी-कभी वेब सर्च में सही उत्तर खोजने से पहले कई पेजों की सर्च करने की आवश्यकता होती है, हम `max_steps` को बढ़ाकर 10 करना पसंद करते हैं।

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    ManagedAgent,
    WebSearchTool,
)

model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
)
```

फिर हम इस एजेंट को एक `ManagedAgent` में रैप करते हैं जो इसे इसके मैनेजर एजेंट द्वारा कॉल करने योग्य बनाएगा।

```py
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

अंत में हम एक मैनेजर एजेंट बनाते हैं, और इनिशियलाइजेशन पर हम अपने मैनेज्ड एजेंट को इसके `managed_agents` आर्गुमेंट में पास करते हैं।

चूंकि यह एजेंट योजना बनाने और सोचने का काम करता है, उन्नत तर्क लाभदायक होगा, इसलिए `CodeAgent` सबसे अच्छा विकल्प होगा।

साथ ही, हम एक ऐसा प्रश्न पूछना चाहते हैं जिसमें वर्तमान वर्ष और अतिरिक्त डेटा गणना शामिल है: इसलिए आइए `additional_authorized_imports=["time", "numpy", "pandas"]` जोड़ें, यदि एजेंट को इन पैकेजों की आवश्यकता हो।

```py
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

बस इतना ही! अब चलिए हमारे सिस्टम को चलाते हैं! हम एक ऐसा प्रश्न चुनते हैं जिसमें गणना और शोध दोनों की आवश्यकता है।

```py
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
```

We get this report as the answer:
```
Based on current growth projections and energy consumption estimates, if LLM trainings continue to scale up at the 
current rhythm until 2030:

1. The electric power required to power the biggest training runs by 2030 would be approximately 303.74 GW, which 
translates to about 2,660,762 GWh/year.

2. Comparing this to countries' electricity consumption:
   - It would be equivalent to about 34% of China's total electricity consumption.
   - It would exceed the total electricity consumption of India (184%), Russia (267%), and Japan (291%).
   - It would be nearly 9 times the electricity consumption of countries like Italy or Mexico.

3. Source of numbers:
   - The initial estimate of 5 GW for future LLM training comes from AWS CEO Matt Garman.
   - The growth projection used a CAGR of 79.80% from market research by Springs.
   - Country electricity consumption data is from the U.S. Energy Information Administration, primarily for the year 
2021.
```

लगता है कि यदि [स्केलिंग हाइपोथिसिस](https://gwern.net/scaling-hypothesis) सत्य बनी रहती है तो हमें कुछ बड़े पावरप्लांट्स की आवश्यकता होगी।

हमारे एजेंट्स ने कार्य को हल करने के लिए कुशलतापूर्वक सहयोग किया! ✅

💡 आप इस ऑर्केस्ट्रेशन को आसानी से अधिक एजेंट्स में विस्तारित कर सकते हैं: एक कोड एक्जीक्यूशन करता है, एक वेब सर्च करता है, एक फाइल लोडिंग को संभालता है।
