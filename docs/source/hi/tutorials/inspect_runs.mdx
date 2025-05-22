# OpenTelemetry के साथ runs का निरीक्षण

[[open-in-colab]]

> [!TIP]
> यदि आप एजेंट्स बनाने में नए हैं, तो पहले [एजेंट्स का परिचय](../conceptual_guides/intro_agents) और [smolagents की गाइडेड टूर](../guided_tour) पढ़ना सुनिश्चित करें।

### Agents runs को लॉग क्यों करें?

Agent runs को डीबग करना जटिल होता है।

यह सत्यापित करना कठिन है कि एक रन ठीक से चला या नहीं, क्योंकि एजेंट वर्कफ़्लो [डिज़ाइन के अनुसार अप्रत्याशित](../conceptual_guides/intro_agents) होते हैं (यदि वे प्रत्याशित होते, तो आप पुराने अच्छे कोड का ही उपयोग कर रहे होते)।

और रन का निरीक्षण करना भी कठिन है: मल्टी-स्टेप एजेंट्स जल्दी ही कंसोल को लॉग से भर देते हैं, और अधिकांश त्रुटियां केवल "LLM dumb" प्रकार की त्रुटियां होती हैं, जिनसे LLM अगले चरण में बेहतर कोड या टूल कॉल लिखकर स्वयं को सुधार लेता है।

इसलिए बाद के निरीक्षण और मॉनिटरिंग के लिए प्रोडक्शन में agent runs को रिकॉर्ड करने के लिए इंस्ट्रुमेंटेशन का उपयोग करना आवश्यक है!

हमने agent runs को इंस्ट्रुमेंट करने के लिए [OpenTelemetry](https://opentelemetry.io/) मानक को अपनाया है।

इसका मतलब है कि आप बस कुछ इंस्ट्रुमेंटेशन कोड चला सकते हैं, फिर अपने एजेंट्स को सामान्य रूप से चला सकते हैं, और सब कुछ आपके प्लेटफॉर्म में लॉग हो जाता है।

यह इस प्रकार होता है:
पहले आवश्यक पैकेज इंस्टॉल करें। यहां हम [Phoenix by Arize AI](https://github.com/Arize-ai/phoenix) इंस्टॉल करते हैं क्योंकि यह लॉग्स को एकत्र और निरीक्षण करने का एक अच्छा समाधान है, लेकिन इस संग्रह और निरीक्षण भाग के लिए आप अन्य OpenTelemetry-कम्पैटिबल प्लेटफॉर्म्स का उपयोग कर सकते हैं।

```shell
pip install smolagents
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

फिर कलेक्टर को बैकग्राउंड में चलाएं।

```shell
python -m phoenix.server.main serve
```

अंत में, अपने एजेंट्स को ट्रेस करने और ट्रेस को नीचे परिभाषित एंडपॉइंट पर Phoenix को भेजने के लिए `SmolagentsInstrumentor` को सेट करें।

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://0.0.0.0:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```
तब आप अपने एजेंट चला सकते हैं!

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    InferenceClientModel,
)

model = InferenceClientModel()

managed_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="managed_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_agent],
)
manager_agent.run(
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
)
```
और फिर आप अपने रन का निरीक्षण करने के लिए `http://0.0.0.0:6006/projects/` पर जा सकते हैं!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/inspect_run_phoenix.png">

आप देख सकते हैं कि CodeAgent ने अपने मैनेज्ड ToolCallingAgent को (वैसे, मैनेज्ड एजेंट एक CodeAgent भी हो सकता था) U.S. 2024 ग्रोथ रेट के लिए वेब सर्च चलाने के लिए कॉल किया। फिर मैनेज्ड एजेंट ने अपनी रिपोर्ट लौटाई और मैनेजर एजेंट ने अर्थव्यवस्था के दोगुना होने का समय गणना करने के लिए उस पर कार्य किया! अच्छा है, है ना?