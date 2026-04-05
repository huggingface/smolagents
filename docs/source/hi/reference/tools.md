# Tools

<Tip warning={true}>

Smolagents एक experimental API है जो किसी भी समय बदल सकता है। एजेंट्स द्वारा लौटाए गए परिणाम भिन्न हो सकते हैं क्योंकि APIs या underlying मॉडल बदलने की संभावना रखते हैं।

</Tip>

एजेंट्स और टूल्स के बारे में अधिक जानने के लिए [introductory guide](../index) पढ़ना सुनिश्चित करें। 
यह पेज underlying क्लासेज के लिए API docs को शामिल करता है।

## Tools

### load_tool

[[autodoc]] load_tool

### tool

[[autodoc]] tool

### Tool

[[autodoc]] Tool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## Default Tools

### PythonInterpreterTool

[[autodoc]] PythonInterpreterTool

### DuckDuckGoSearchTool

[[autodoc]] DuckDuckGoSearchTool

### VisitWebpageTool

[[autodoc]] VisitWebpageTool

### UserInputTool

[[autodoc]] UserInputTool

## ToolCollection

[[autodoc]] ToolCollection

## Agent टाइप्स

एजेंट्स टूल्स के बीच किसी भी प्रकार की ऑब्जेक्ट को संभाल सकते हैं; टूल्स, पूरी तरह से मल्टीमोडल होने के कारण, टेक्स्ट, इमेज, ऑडियो, वीडियो सहित अन्य प्रकारों को स्वीकार और रिटर्न कर सकते हैं। 
टूल्स के बीच अनुकूलता बढ़ाने के साथ-साथ इन रिटर्न्स को ipython (jupyter, colab, ipython notebooks, ...) में सही ढंग से रेंडर करने के लिए, हम इन टाइप्स के आसपास रैपर क्लासेज को लागू करते हैं।

रैप किए गए ऑब्जेक्ट्स को प्रारंभ में जैसा व्यवहार करना चाहिए वैसा ही करना जारी रखना चाहिए; एक टेक्स्ट ऑब्जेक्ट को अभी भी स्ट्रिंग की तरह व्यवहार करना चाहिए|
एक इमेज ऑब्जेक्ट को अभी भी `PIL.Image` की तरह व्यवहार करना चाहिए।

इन टाइप्स के तीन विशिष्ट उद्देश्य हैं:

- टाइप पर `to_raw` को कॉल करने से अंतर्निहित ऑब्जेक्ट रिटर्न होना चाहिए
- टाइप पर `to_string` को कॉल करने से ऑब्जेक्ट को स्ट्रिंग के रूप में रिटर्न होना चाहिए: वह `AgentText` के मामले में स्ट्रिंग हो सकती है लेकिन अन्य उदाहरणों में ऑब्जेक्ट के सीरियलाइज्ड वर्जन का पाथ होगा
- इसे एक ipython kernel में प्रदर्शित करने पर ऑब्जेक्ट को सही ढंग से प्रदर्शित करना चाहिए

### AgentText

[[autodoc]] smolagents.agent_types.AgentText

### AgentImage

[[autodoc]] smolagents.agent_types.AgentImage

### AgentAudio

[[autodoc]] smolagents.agent_types.AgentAudio
