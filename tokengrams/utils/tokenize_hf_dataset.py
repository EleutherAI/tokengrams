import numpy as np
from typing import Generator
from argparse import ArgumentParser
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from typing import Union
import os
import multiprocessing as mp
from tqdm import tqdm

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
        # Concatenate all available splits in the DatasetDict
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
    tokenizer: AutoTokenizer,
    output_prefix: str,
    text_key: str = "text",
    batch_size: int = 1000,
    buffer_size: int = 10_000_000,
    eos_token: int | None = None,
    num_workers: int = 4
):
    mmap = np.memmap(f'{output_prefix}.bin', dtype=np.uint16, mode='w+', shape=(buffer_size,))

    total_tokens = 0
    pool = mp.Pool(num_workers)

    pbar = tqdm(desc="Tokenizing")
    for batch in data:
        tokenize_args = [(batch, tokenizer, text_key, eos_token)]
        new_tokens = pool.map(tokenize_batch, tokenize_args)[0]

        if total_tokens + len(new_tokens) > mmap.shape[0]:
            mmap = np.memmap(f'{output_prefix}.bin', dtype=np.uint16, mode='r+', shape=(mmap.shape[0] * 2,))

        mmap[total_tokens:total_tokens + len(new_tokens)] = new_tokens
        total_tokens += len(new_tokens)
        pbar.update(batch_size)

    pool.close()
    pool.join()

    # Resize mmap to actual size
    with open(f'{output_prefix}.bin', 'r+b') as f:
        f.truncate(total_tokens * np.uint16().itemsize)

    mmap = np.memmap(f'{output_prefix}.bin', dtype=np.uint16, mode='r+', shape=(total_tokens,))
    mmap.flush()

    pbar.close()
    return total_tokens

def get_args(input_args=None):
    parser = ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the Hugging Face dataset to use",
    )
    group.add_argument(
        "--config_name",
        type=str,
        default=None
    )
    group.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the Hugging Face dataset",
    )
    group.add_argument(
        "--stream",
        action="store_true",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name or path of the pre-trained tokenizer to use",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    args = parser.parse_args(input_args)
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args

def main(input_args=None):
    args = get_args(input_args)
    batch_size = 10_000

    # Get tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name) # type: ignore
    eos_token = tokenizer.eos_token_id if args.append_eod else None # type: ignore

    # Get data
    os.makedirs('.cache', exist_ok=True)
    dataset = load_dataset(
        args.dataset_name, 
        args.config_name,
        cache_dir=os.path.join(os.getcwd(), '.cache'),
        split=args.split,
        streaming=args.stream,
    )

    data = get_dataset_iterator(dataset, batch_size)
    
    # Tokenize and save as memory-mapped array
    total_tokens = tokenize_and_write_mmap(
        data, 
        tokenizer, 
        args.output_prefix, 
        eos_token=eos_token,
        batch_size=batch_size,
        num_workers=args.workers
    )
    
    print(f"{total_tokens} tokens saved as memory-mapped array in {args.output_prefix}.bin")

if __name__ == "__main__":
    main()