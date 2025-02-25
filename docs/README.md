<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Generating the documentation

To generate the documentation, you have to build it. Several packages are necessary to build the doc.

First, you need to install the project itself by running the following command at the root of the code repository:

```bash
pip install -e .
```

You also need to install 2 extra packages:

```bash
# `hf-doc-builder` to build the docs
pip install git+https://github.com/huggingface/doc-builder@main
# `watchdog` for live reloads
pip install watchdog
```

---
**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to commit the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages with the pip install command above,
you can generate the documentation by typing the following command:

```bash
doc-builder build smolagents docs/source/en/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

## Previewing the documentation

To preview the docs, run the following command:

```bash
doc-builder preview smolagents docs/source/en/
```

The docs will be viewable at [http://localhost:5173](http://localhost:5173). You can also preview the docs once you
have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

---
**NOTE**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update
`_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

---

## Adding a new element to the navigation bar

Accepted files are Markdown (.md).

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/smolagents/blob/main/docs/source/_toctree.yml) file.

## Renaming section headers and moving sections

It helps to keep the old links working when renaming the section header and/or moving sections from one document to another. This is because the old links are likely to be used in Issues, Forums, and Social media and it'd make for a much more superior user experience if users reading those months later could still easily navigate to the originally intended information.

Therefore, we simply keep a little map of moved sections at the end of the document where the original section was. The key is to preserve the original anchor.

So if you renamed a section from: "Section A" to "Section B", then you can add at the end of the file:

```
Sections that were moved:

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```
and of course, if you moved it to another file, then:

```
Sections that were moved:

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

Use the relative style to link to the new file so that the versioned docs continue to work.

For an example of a rich moved section set please see the very end of [the transformers Trainer doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md).


## Writing Documentation - Specification

The `huggingface/smolagents` documentation follows the
[Google documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) style for docstrings,
although we can write them directly in Markdown.

### Adding a new tutorial

Adding a new tutorial or section is done in two steps:

- Add a new Markdown (.md) file under `./source`.
- Link that file in `./source/_toctree.yml` on the correct toc-tree.

Make sure to put your new file under the proper section. If you have a doubt, feel free to ask in a Github Issue or PR.

### Writing source documentation

Values that should be put in `code` should either be surrounded by backticks: \`like so\`. Note that argument names
and objects like True, None, or any strings should usually be put in `code`.

When mentioning a class, function, or method, it is recommended to use our syntax for internal links so that our tool
adds a link to its documentation with this syntax: \[\`XXXClass\`\] or \[\`function\`\]. This requires the class or
function to be in the main package.

If you want to create a link to some internal class or function, you need to
provide its path. For instance: \[\`utils.ModelOutput\`\]. This will be converted into a link with
`utils.ModelOutput` in the description. To get rid of the path and only keep the name of the object you are
linking to in the description, add a ~: \[\`~utils.ModelOutput\`\] will generate a link with `ModelOutput` in the description.

The same works for methods so you can either use \[\`XXXClass.method\`\] or \[~\`XXXClass.method\`\].

#### Defining arguments in a method

Arguments should be defined with the `Args:` (or `Arguments:` or `Parameters:`) prefix, followed by a line return and
an indentation. The argument should be followed by its type, with its shape if it is a tensor, a colon, and its
description:

```
    Args:
        n_layers (`int`): The number of layers of the model.
```

If the description is too long to fit in one line, another indentation is necessary before writing the description
after the argument.

Here's an example showcasing everything so far:

```
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AlbertTokenizer`]. See [`~PreTrainedTokenizer.encode`] and
            [`~PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
```

For optional arguments or arguments with defaults we follow the following syntax: imagine we have a function with the
following signature:

```
def my_function(x: str = None, a: float = 1):
```

then its documentation should look like this:

```
    Args:
        x (`str`, *optional*):
            This argument controls ...
        a (`float`, *optional*, defaults to 1):
            This argument is used to ...
```

Note that we always omit the "defaults to \`None\`" when None is the default for any argument. Also note that even
if the first line describing your argument type and its default gets long, you can't break it on several lines. You can
however write as many lines as you want in the indented description (see the example above with `input_ids`).

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done between two lines of three backticks as usual in Markdown:


````
```
# first line of code
# second line
# etc
```
````

#### Writing a return block

The return block should be introduced with the `Returns:` prefix, followed by a line return and an indentation.
The first line should be the type of the return, followed by a line return. No need to indent further for the elements
building the return.

Here's an example of a single value return:

```
    Returns:
        `List[int]`: A list of integers in the range [0, 1] --- 1 for a special token, 0 for a sequence token.
```

Here's an example of a tuple return, comprising several objects:

```
    Returns:
        `tuple(torch.FloatTensor)` comprising various elements depending on the configuration ([`BertConfig`]) and inputs:
        - ** loss** (*optional*, returned when `masked_lm_labels` is provided) `torch.FloatTensor` of shape `(1,)` --
          Total loss is the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        - **prediction_scores** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) --
          Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
```

#### Adding an image

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.

#### Writing documentation examples

The syntax for Example docstrings can look as follows:

```
    Example:

    ```python
    >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'
    ```
```

The docstring should give a minimal, clear example of how the respective model
is to be used in inference and also include the expected (ideally sensible)
output.
Often, readers will try out the example before even going through the function
or class definitions. Therefore, it is of utmost importance that the example
works as expected.

