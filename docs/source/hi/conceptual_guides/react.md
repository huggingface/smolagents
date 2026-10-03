# मल्टी-स्टेप एजेंट्स कैसे काम करते हैं?

ReAct फ्रेमवर्क ([Yao et al., 2022](https://huggingface.co/papers/2210.03629)) वर्तमान में एजेंट्स बनाने का मुख्य दृष्टिकोण है।

नाम दो शब्दों, "Reason" (तर्क) और "Act" (क्रिया) के संयोजन पर आधारित है। वास्तव में, इस आर्किटेक्चर का पालन करने वाले एजेंट अपने कार्य को उतने चरणों में हल करेंगे जितने आवश्यक हों, प्रत्येक चरण में एक Reasoning कदम होगा, फिर एक Action कदम होगा, जहाँ यह टूल कॉल्स तैयार करेगा जो उसे कार्य को हल करने के करीब ले जाएंगे।

ReAct प्रक्रिया में पिछले चरणों की मेमोरी रखना शामिल है।

> [!TIP]
> मल्टी-स्टेप एजेंट्स के बारे में अधिक जानने के लिए [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) ब्लॉग पोस्ट पढ़ें।

यहाँ एक वीडियो ओवरव्यू है कि यह कैसे काम करता है:

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

![ReAct एजेंट का फ्रेमवर्क](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

हम दो प्रकार के ToolCallingAgent को लागू करते हैं:
- [`ToolCallingAgent`] अपने आउटपुट में टूल कॉल को JSON के रूप में जनरेट करता है।
- [`CodeAgent`] ToolCallingAgent का एक नया प्रकार है जो अपने टूल कॉल को कोड के ब्लॉब्स के रूप में जनरेट करता है, जो उन LLM के लिए वास्तव में अच्छी तरह काम करता है जिनका कोडिंग प्रदर्शन मजबूत है।
