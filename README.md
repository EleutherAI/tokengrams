# Tokengrams
Tokengrams allows you to efficiently compute $n$-gram statistics for pre-tokenized text corpora used to train large language models. It does this not by explicitly pre-computing the $n$-gram counts for fixed $n$, but by creating a [suffix array](https://en.wikipedia.org/wiki/Suffix_array) index which allows you to efficiently compute the count of an $n$-gram on the fly for any $n$.

Our code also allows you to turn your suffix array index into an efficient $n$-gram language model, which can be used to generate text or compute the perplexity of a given text.

The backend is written in Rust, and the Python bindings are generated using [PyO3](https://github.com/PyO3/pyo3).

# Installation

```bash
pip install tokengrams
```

# Development

```bash
pip install maturin
maturin develop
```

# Usage

## Building an index
```python
from tokengrams import MemmapIndex

# Create a new index from an on-disk corpus called `document.bin` and save it to
# `pile.idx`.
index = MemmapIndex.build(
    "/data/document.bin",
    "/pile.idx",
)

# Verify index correctness
print(index.is_sorted())
  
# Get the count of "hello world" in the corpus.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
print(index.count(tokenizer.encode("hello world")))

# You can now load the index from disk later using __init__
index = MemmapIndex(
    "/data/document.bin",
    "/pile.idx"
)
```

## Using an index

```python
# Count how often each token in the corpus succeeds "hello world".
print(index.count_next(tokenizer.encode("hello world")))

# Parallelise over queries
print(index.batch_count_next(
    [tokenizer.encode("hello world"), tokenizer.encode("hello universe")]
))

# Autoregressively sample 10 tokens using 5-gram language statistics. Initial
# gram statistics are derived from the query, with lower order gram statistics used 
# until the sequence contains at least 5 tokens.
print(index.sample(tokenizer.encode("hello world"), n=5, k=10))

# Parallelize over sequence generations
print(index.batch_sample(tokenizer.encode("hello world"), n=5, k=10, num_samples=20))

# Query whether the corpus contains "hello world"
print(index.contains(tokenizer.encode("hello world")))

# Get all n-grams beginning with "hello world" in the corpus
print(index.positions(tokenizer.encode("hello world")))
```

# Support

The best way to get support is to open an issue on this repo or post in #inductive-biases in the [EleutherAI Discord server](https://discord.gg/eleutherai). If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!