from itertools import pairwise
from tempfile import NamedTemporaryFile
import random
from tokengrams import MemmapIndex, ShardedMemmapIndex

import numpy as np


def check_sharded_index(index: ShardedMemmapIndex, tokens: list[int], eos_token: int):
    # Check unigram counts
    for t in tokens:
        assert index.contains([t]) == (t in tokens)
        assert index.count([t]) == tokens.count(t)

    # Check bigram counts
    bigrams = list(pairwise(tokens))
    for b in bigrams:
        if not eos_token in b:
            assert index.contains(list(b)) == (b in bigrams)
            assert index.count(list(b)) == bigrams.count(b)

    # Check bigram samples
    for i in range(len(tokens[:20])):
        query = tokens[:i]
        sample = index.sample_unsmoothed(query, 2, 1, 1)[0]
        assert len(sample) == 1 + len(query)
        assert all(s in tokens for s in sample)


def test_sharded_index():
    tokens = [random.randint(0, 2**16 - 1) for _ in range(10_000)]

    eos_token = 0
    mid = len(tokens) // 2
    chunked_tokens = tokens[:mid] + [eos_token] + tokens[mid:] + [eos_token]

    with NamedTemporaryFile() as token_file_1, NamedTemporaryFile() as index_file_1, \
         NamedTemporaryFile() as token_file_2, NamedTemporaryFile() as index_file_2:

        shard_files = [
            (token_file_1.name, index_file_1.name), 
            (token_file_2.name, index_file_2.name)
        ]

        token_file_1.write(np.array(chunked_tokens[:mid + 1], dtype=np.uint16).tobytes())
        token_file_1.flush()
        token_file_2.write(np.array(chunked_tokens[mid + 1:], dtype=np.uint16).tobytes())
        token_file_2.flush()
        
        for token_file, index_file in shard_files:
            MemmapIndex.build(token_file, index_file)
        
        index = ShardedMemmapIndex(shard_files)
        check_sharded_index(index, chunked_tokens, eos_token)