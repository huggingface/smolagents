# एजेंटिक RAG

[[open-in-colab]]

रिट्रीवल-ऑगमेंटेड-जनरेशन (RAG) है "एक यूजर के प्रश्न का उत्तर देने के लिए LLM का उपयोग करना, लेकिन उत्तर को एक नॉलेज बेस से प्राप्त जानकारी पर आधारित करना"। इसमें वैनिला या फाइन-ट्यून्ड LLM का उपयोग करने की तुलना में कई फायदे हैं: कुछ नाम लेने के लिए, यह उत्तर को सत्य तथ्यों पर आधारित करने और काल्पनिक बातों को कम करने की अनुमति देता है, यह LLM को डोमेन-विशिष्ट ज्ञान प्रदान करने की अनुमति देता है, और यह नॉलेज बेस से जानकारी तक पहुंच का सूक्ष्म नियंत्रण प्रदान करता है।

लेकिन वैनिला RAG की सीमाएं हैं, सबसे महत्वपूर्ण ये दो:
- यह केवल एक रिट्रीवल स्टेप करता है: यदि परिणाम खराब हैं, तो जनरेशन भी बदले में खराब होगा।
- सिमेंटिक समानता की गणना यूजर के प्रश्न को संदर्भ के रूप में करके की जाती है, जो अनुकूल नहीं हो सकती: उदाहरण के लिए, यूजर का प्रश्न अक्सर एक सवाल होगा, जबकि सही उत्तर देने वाला डॉक्यूमेंट सकारात्मक स्वर में हो सकता है, और इसका समानता स्कोर अन्य स्रोत दस्तावेज़ों की तुलना में कम हो सकता है, जो प्रश्नवाचक स्वर में हो सकते हैं। इससे संबंधित जानकारी को चूकने का जोखिम होता है।

हम एक RAG एजेंट बनाकर इन समस्याओं को कम कर सकते हैं: बहुत सरल तरीके से, एक रिट्रीवर टूल से लैस एजेंट!

यह एजेंट करेगा: ✅ स्वयं क्वेरी तैयार करेगा और ✅ आवश्यकता पड़ने पर पुनः-प्राप्ति के लिए समीक्षा करेगा।

इसलिए यह सहज रूप से कुछ उन्नत RAG तकनीकों को प्राप्त कर लेना चाहिए!
- सिमेंटिक खोज में सीधे यूजर क्वेरी का संदर्भ के रूप में उपयोग करने के बजाय, एजेंट स्वयं एक संदर्भ वाक्य तैयार करता है जो लक्षित डॉक्यूमेंट्स के करीब हो सकता है, जैसा कि [HyDE](https://huggingface.co/papers/2212.10496) में किया गया है।
एजेंट जनरेट किए गए स्निपेट्स का उपयोग कर सकता है और आवश्यकता पड़ने पर पुनः-प्राप्ति कर सकता है, जैसा कि [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/) में किया गया है।

चलिए इस सिस्टम को बनाते हैं। 🛠️

आवश्यक डिपेंडेंसी इंस्टॉल करने के लिए नीचे दी गई लाइन चलाएं।
```bash
!pip install smolagents pandas langchain langchain-community sentence-transformers rank_bm25 --upgrade -q
```
HF Inference API को कॉल करने के लिए, आपको अपने एनवायरनमेंट वेरिएबल `HF_TOKEN` के रूप में एक वैध टोकन की आवश्यकता होगी।
हम इसे लोड करने के लिए python-dotenv का उपयोग करते हैं।
```py
from dotenv import load_dotenv
load_dotenv()
```

हम पहले एक नॉलेज बेस लोड करते हैं जिस पर हम RAG को लागू करना चाहते हैं: यह डेटा सेट Hugging Face के कई लाइब्रेरी के डॉक्यूमेंट पृष्ठों का संकलन है, जिन्हें Markdown में स्टोर किया गया है। हम केवल `transformers` लाइब्रेरी के दस्तावेज़ों को रखेंगे।

फिर डेटासेट को प्रोसेस करके और इसे एक वेक्टर डेटाबेस में स्टोर करके नॉलेज बेस तैयार करें जिसे रिट्रीवर द्वारा उपयोग किया जाएगा।

हम [LangChain](https://python.langchain.com/docs/introduction/) का उपयोग करते हैं क्योंकि इसमें उत्कृष्ट वेक्टर डेटाबेस उपयोगिताएं हैं।

```py
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)
```

अब डॉक्यूमेंट्स तैयार हैं।

तो चलिए अपना एजेंटिक RAG सिस्टम बनाएं!

👉 हमें केवल एक RetrieverTool की आवश्यकता है जिसका उपयोग हमारा एजेंट नॉलेज बेस से जानकारी प्राप्त करने के लिए कर सकता है।

चूंकि हमें टूल के एट्रीब्यूट के रूप में एक vectordb जोड़ने की आवश्यकता है, हम सरल टूल कंस्ट्रक्टर को `@tool` डेकोरेटर के साथ सीधे उपयोग नहीं कर सकते: इसलिए हम [tools tutorial](../tutorials/tools) में हाइलाइट किए गए सेटअप का पालन करेंगे।

```py
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```
हमने BM25 का उपयोग किया है, जो एक क्लासिक रिट्रीवल विधि है,  क्योंकि इसे सेटअप करना बहुत आसान है।
रिट्रीवल सटीकता में सुधार करने के लिए, आप BM25 को डॉक्यूमेंट्स के लिए वेक्टर प्रतिनिधित्व का उपयोग करके सिमेंटिक खोज से बदल सकते हैं: इस प्रकार आप एक अच्छा एम्बेडिंग मॉडल चुनने के लिए [MTEB Leaderboard](https://huggingface.co/mteb) पर जा सकते हैं।

अब यह सीधा है कि एक एजेंट बनाया जाए जो इस `retriever_tool` का उपयोग करेगा!


एजेंट को इनिशियलाइजेशन पर इन आर्गुमेंट्स की आवश्यकता होगी:
- `tools`: टूल्स की एक सूची जिन्हें एजेंट कॉल कर सकेगा।
- `model`: LLM जो एजेंट को पावर देता है।
हमारा `model` एक कॉलेबल होना चाहिए जो इनपुट के रूप में संदेशों की एक सूची लेता है और टेक्स्ट लौटाता है। इसे एक stop_sequences आर्गुमेंट भी स्वीकार करने की आवश्यकता है जो बताता है कि जनरेशन कब रोकनी है। सुविधा के लिए, हम सीधे पैकेज में प्रदान की गई HfEngine क्लास का उपयोग करते हैं ताकि एक LLM इंजन मिल सके जो Hugging Face के Inference API को कॉल करता है।

और हम [meta-llama/Llama-3.3-70B-Instruct](meta-llama/Llama-3.3-70B-Instruct) का उपयोग llm इंजन के रूप में करते हैं क्योंकि:
- इसमें लंबा 128k कॉन्टेक्स्ट है, जो लंबे स्रोत दस्तावेजों को प्रोसेस करने में मददगार है
- यह हर समय HF के Inference API पर मुफ्त में उपलब्ध है!

_नोट:_ Inference API विभिन्न मानदंडों के आधार पर मॉडल होस्ट करता है, और डिप्लॉय किए गए मॉडल बिना पूर्व सूचना के अपडेट या बदले जा सकते हैं। इसके बारे में अधिक जानें [यहां](https://huggingface.co/docs/api-inference/supported-models) पढ़ें।

```py
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(
    tools=[retriever_tool], model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct"), max_steps=4, verbosity_level=2
)
```

CodeAgent को इनिशियलाइज करने पर, इसे स्वचालित रूप से एक डिफ़ॉल्ट सिस्टम प्रॉम्प्ट दिया गया है जो LLM इंजन को चरण-दर-चरण प्रोसेस करने और कोड स्निपेट्स के रूप में टूल कॉल जनरेट करने के लिए कहता है, लेकिन आप आवश्यकतानुसार इस प्रॉम्प्ट टेम्पलेट को अपने से बदल सकते हैं।

जब CodeAgent का `.run()` मेथड लॉन्च किया जाता है, तो एजेंट LLM इंजन को कॉल करने का कार्य करता है, और टूल कॉल्स को निष्पादित करता है, यह सब एक लूप में होता है, जो तब तक चलता है जब तक टूल final_answer के साथ अंतिम उत्तर के रूप में नहीं बुलाया जाता।

```py
agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

print("Final output:")
print(agent_output)
```


