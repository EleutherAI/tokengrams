# Tokengrams
Tokengrams allows you to efficiently compute $n$-gram statistics for pre-tokenized text corpora used to train large language models. It does this not by explicitly pre-computing the $n$-gram counts for fixed $n$, but by creating a [suffix array](https://en.wikipedia.org/wiki/Suffix_array) index which allows you to efficiently compute the count of an $n$-gram on the fly for any $n$.

Our code also allows you to turn your suffix array index into an efficient $n$-gram language model, which can be used to generate text or compute the perplexity of a given text.

The backend is written in Rust, and the Python bindings are generated using [PyO3](https://github.com/PyO3/pyo3).

# Installation

```bash
pip install tokengrams
```

# Usage

[Full text generation demo](https://colab.research.google.com/drive/1CEHoIjLboGl8YPbIqnWJlPYMm1wVOrrj?usp=sharing)

## Preparing data

Use a dataset of u16 or u32 tokens, or prepare one from a HuggingFace dataset.

```python
# Get pre-tokenized dataset
from huggingface_hub import HfApi, hf_hub_download

hf_hub_download(
  repo_id="EleutherAI/pile-standard-pythia-preshuffled", 
  repo_type="dataset", 
  filename="document-00000-of-00020.bin", 
  local_dir="."
)
```
```python
# Tokenize HF dataset
from tokengrams import tokenize_hf_dataset
from datasets import load_dataset
from transformers import AutoTokenizer

tokenize_hf_dataset(
    dataset=load_dataset("EleutherAI/lambada_openai", "en"),
    tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"),
    output_path="lambada.bin",
    text_key="text",
    append_eod=True,
    workers=1,
)
```

## Building an index
```python
from tokengrams import MemmapIndex

# Create a new index from an on-disk corpus of u16 tokens and save it to a .idx file. 
# Set verbose to true to include a progress bar for the index sort.
index = MemmapIndex.build(
    "document-00000-of-00020.bin",
    "document-00000-of-00020.idx",
    vocab=2**16,
    verbose=True
)

# True for any valid index.
print(index.is_sorted())
  
# Get the count of "hello world" in the corpus.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
print(index.count(tokenizer.encode("hello world")))

# You can now load the index from disk later using __init__
index = MemmapIndex(
    "document-00000-of-00020.bin",
    "document-00000-of-00020.idx",
    vocab=2**16
)
```

## Using an index

```python
# Count how often each token in the corpus succeeds "hello world".
print(index.count_next(tokenizer.encode("hello world")))
print(index.batch_count_next(
    [tokenizer.encode("hello world"), tokenizer.encode("hello universe")]
))

# Get smoothed probabilities for query continuations
print(index.smoothed_probs(tokenizer.encode("hello world")))
print(index.batch_smoothed_probs(
    [tokenizer.encode("hello world"), tokenizer.encode("hello universe")]
))

# Autoregressively sample 10 tokens using 5-gram language statistics. Initial
# gram statistics are derived from the query, with lower order gram statistics used 
# until the sequence contains at least 5 tokens.
print(index.sample_unsmoothed(tokenizer.encode("hello world"), n=5, k=10, num_samples=20))
print(index.sample_smoothed(tokenizer.encode("hello world"), n=5, k=10, num_samples=20))

# Query whether the corpus contains "hello world"
print(index.contains(tokenizer.encode("hello world")))

# Get all n-grams beginning with "hello world" in the corpus
print(index.positions(tokenizer.encode("hello world")))
```

## Scaling

Corpora small enough to fit in memory can use an InMemoryIndex:

```python
from tokengrams import InMemoryIndex

tokens = [0, 1, 2, 3, 4]
index = InMemoryIndex(tokens, vocab=5)
```

Larger corpora must use a MemmapIndex.

Some systems struggle with memory mapping extremely large tables (e.g. 40 billion tokens), causing unexpected bus errors. To prevent this split the corpus into shards then use a ShardedMemmapIndex to sort and query the table shard by shard:

```python
from tokengrams import ShardedMemmapIndex
from huggingface_hub import HfApi, hf_hub_download

files = [
    file for file in HfApi().list_repo_files("EleutherAI/pile-standard-pythia-preshuffled", repo_type="dataset")
    if file.endswith('.bin')
]

index_paths = []
for file in files:
    hf_hub_download("EleutherAI/pile-standard-pythia-preshuffled", repo_type="dataset", filename=file, local_dir=".")
    index_paths.append((file, f'{file.rstrip(".bin")}.idx'))

index = ShardedMemmapIndex.build(index_paths, vocab=2**16, verbose=True)
```
### Tokens

Tokengrams builds indices from on-disk corpora of either u16 or u32 tokens, supporting a maximum vocabulary size of 2<sup>32</sup>. In practice, however, vocabulary size is limited by the length of the largest word size vector the machine can allocate in memory. 

Corpora with vocabulary sizes smaller than 2<sup>16</sup> must use u16 tokens.

## Performance

Index build times for in-memory corpora scale inversely with the number of available CPU threads, whereas if the index reads from or writes to a file it is likely to be IO bound.

The time complexities of count_next(query) and sample_unsmoothed(query) are O(n log n), where n is ~ the number of completions for the query. The time complexity of sample_smoothed(query) is O(m n log n) where m is the n-gram order.

<table>
  <tr>
    <td><img src="./tokengrams/benchmark/MemmapIndex_build_times.png" alt="Sample build times for an IO bound index"></td>
    <td><img src="./tokengrams/benchmark/MemmapIndex_count_next_times.png" alt="Sample count_next times for an IO bound index"></td>
  </tr>
</table>

# Development

```bash
cargo build
cargo test
```

Develop Python bindings:

```bash
pip install maturin
maturin develop
pytest
```

# Support

The best way to get support is to open an issue on this repo or post in #interp-across-time in the [EleutherAI Discord server](https://discord.gg/eleutherai). If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!
