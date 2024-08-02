from itertools import pairwise
from tempfile import NamedTemporaryFile
import random
from tokengrams import MemmapIndexU16, MemmapIndexU32

import numpy as np


def check_index(index: MemmapIndexU32 | MemmapIndexU16, tokens: list[int]):
    assert index.is_sorted()

    # Check unigram counts
    for t in tokens:
        assert index.contains([t]) == (t in tokens)
        assert index.count([t]) == tokens.count(t)

    # Check bigram counts
    bigrams = list(pairwise(tokens))
    for b in bigrams:
        assert index.contains(list(b)) == (b in bigrams)
        assert index.count(list(b)) == bigrams.count(b)


def test_typed_index():
    tokens = [random.randint(0, 2**16 - 1) for _ in range(10_000)]

    with NamedTemporaryFile() as token_file_u16, NamedTemporaryFile() as index_file_1, \
         NamedTemporaryFile() as token_file_u32, NamedTemporaryFile() as index_file_2:

        token_file_u16.write(np.array(tokens, dtype=np.uint16).tobytes())
        token_file_u16.flush()
        token_file_u32.write(np.array(tokens, dtype=np.uint32).tobytes())
        token_file_u32.flush()
        
        first = MemmapIndexU16.build(token_file_u16.name, index_file_1.name, False)
        second = MemmapIndexU32.build(token_file_u32.name, index_file_2.name, False)

        check_index(first, tokens)
        check_index(second, tokens)