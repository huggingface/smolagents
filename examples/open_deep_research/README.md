# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)!

Read more about this implementation's goal and methods [in our blog post](https://huggingface.co/blog/open-deep-research).

This agent achieves 55% pass@1 on GAIA validation set, vs 67% for Deep Research.

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install smolagents[dev]
```

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```

<b>Note</b>: It needs the usage tier 5, which can be accessed to call the model 'o1'.

You can check it at the bottom of [OpenAI platform](https://platform.openai.com/settings/organization/limits).
