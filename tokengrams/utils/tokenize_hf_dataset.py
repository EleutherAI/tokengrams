import multiprocessing as mp
from typing import Union, Generator
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm


def tokenize_hf_dataset(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    output_path: Path,
    text_key="text",
    append_eod: bool = False,
    workers: int = 1,
):
    batch_size = 10_000
    eos_token = tokenizer.eos_token_id if append_eod else None
    vocab_size = get_vocab_size(tokenizer)
    if vocab_size > 2**32:
        raise ValueError(f"Tokenizer vocab size {vocab_size} is too large for uint32")

    data = get_dataset_iterator(dataset, batch_size)

    # Tokenize and save as memory-mapped array
    total_tokens = tokenize_and_write_mmap(
        data, 
        tokenizer, 
        output_path, 
        eos_token=eos_token,
        text_key=text_key,
        num_workers=workers,
        dtype=np.dtype(np.uint16 if vocab_size < 2**16 else np.uint32)
    )
    print(f"{total_tokens} tokens saved to {output_path}")

def get_vocab_size(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> int:
    """Get the vocab size of the tokenizer."""
    if hasattr(tokenizer, 'vocab_size'):
        return tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):
        return len(tokenizer.get_vocab())
    else:
        return len(tokenizer)

def get_dataset_iterator(data: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict], batch_size: int):
    """Get an iterator for the dataset, handling different dataset types."""
    if isinstance(data, IterableDataset):
        return iter(data.iter(batch_size=batch_size))
    elif isinstance(data, Dataset):
        return (
            data.select(range(i, min(i + batch_size, len(data))))
            for i in range(0, len(data), batch_size)
        )
    elif isinstance(data, DatasetDict) or isinstance(data, IterableDatasetDict):
        # Concatenate all available splits
        concatenated_dataset = concatenate_datasets(list(data.values()))
        return concatenated_dataset.iter(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported dataset type: {type(data)}")

def tokenize_batch(args):
    batch, tokenizer, text_key, eos_token = args
    tokenized = tokenizer(batch[text_key], add_special_tokens=False, truncation=False, padding=False)
    suffix = [eos_token] if eos_token is not None else []
    
    all_tokens = []
    for tokens in tokenized['input_ids']:
        all_tokens.extend(tokens)
        all_tokens.extend(suffix)
    return all_tokens

def tokenize_and_write_mmap(
    data: Generator,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    output_path: Path,
    text_key: str = "text",
    buffer_size: int = 10_000_000,
    eos_token: int | None = None,
    num_workers: int = 4,
    dtype: np.dtype = np.dtype(np.uint16)
):
    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=(buffer_size,))

    total_tokens = 0
    pool = mp.Pool(num_workers)

    pbar = tqdm(desc="Tokenizing")
    for batch in data:
        tokenize_args = [(batch, tokenizer, text_key, eos_token)]
        new_tokens = pool.map(tokenize_batch, tokenize_args)[0]

        if total_tokens + len(new_tokens) > mmap.shape[0]:
            mmap = np.memmap(output_path, dtype=dtype, mode='r+', shape=(mmap.shape[0] * 2,))

        mmap[total_tokens:total_tokens + len(new_tokens)] = new_tokens
        total_tokens += len(new_tokens)
        pbar.update(len(batch))

    pool.close()
    pool.join()

    # Resize mmap to actual size
    with open(output_path, 'r+b') as f:
        f.truncate(total_tokens * dtype.itemsize)

    mmap = np.memmap(output_path, dtype=dtype, mode='r+', shape=(total_tokens,))
    mmap.flush()

    pbar.close()
    return total_tokens