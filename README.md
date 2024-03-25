# Tokengrams
This library allows you to efficiently compute $n$-gram statistics for pre-tokenized text corpora used to train large language models. It does this not by explicitly pre-computing the $n$-gram counts for fixed $n$, but by creating a [suffix array](https://en.wikipedia.org/wiki/Suffix_array) index which allows you to efficiently compute the count of an $n$-gram on the fly for any $n$.

Our code also allows you to turn your suffix array index into an efficient $n$-gram language model, which can be used to generate text or compute the perplexity of a given text.

The backend is written in Rust, and the Python bindings are generated using [PyO3](https://github.com/PyO3/pyo3).

# Installation
Currently you need to build and install from source using `maturin`. We plan to release wheels on PyPI soon.

```bash
pip install maturin
maturin develop
```

# Usage
```python
from tokengrams import MemmapIndex

# Create a new index from an on-disk corpus called `document.bin` and save it to
# `pile.idx`
index = MemmapIndex.build(
    "/mnt/ssd-1/pile_preshuffled/standard/document.bin",
    "/mnt/ssd-1/nora/pile.idx",
)

# Get the count of "hello world" in the corpus
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
print(index.count(tokenizer.encode("hello world")))

# You can now load the index from disk later using __init__
index = MemmapIndex(
    "/mnt/ssd-1/pile_preshuffled/standard/document.bin",
    "/mnt/ssd-1/nora/pile.idx"
)
```