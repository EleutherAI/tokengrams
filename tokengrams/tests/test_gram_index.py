from itertools import pairwise
from tempfile import NamedTemporaryFile

from tokengrams import GramIndex
from hypothesis import given, strategies as st

import numpy as np


def check_gram_index(index: GramIndex, tokens: list[int]):
    # Check unigram counts
    for t in tokens:
        assert index.count([t]) == tokens.count(t)

    # Check bigram counts
    bigrams = list(pairwise(tokens))
    for b in bigrams:
        assert index.count(list(b)) == bigrams.count(b)

@given(
    st.lists(
        st.integers(0, 2 ** 16 - 1), min_size=1,
    )
)
def test_gram_index(tokens: list[int]):
    # Construct index
    index = GramIndex(tokens)
    check_gram_index(index, tokens)

    # Save to disk and check that we can load it back
    with NamedTemporaryFile() as f:
        memmap = np.memmap(f, dtype=np.uint16, mode="w+", shape=(len(tokens),))
        memmap[:] = tokens

        index = GramIndex.from_token_file(f.name, None)
        check_gram_index(index, tokens)

        # Now check limited token loading
        for limit in range(1, len(tokens) + 1):
            index = GramIndex.from_token_file(f.name, limit)
            check_gram_index(index, tokens[:limit])
