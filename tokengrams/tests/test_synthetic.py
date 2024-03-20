from tokengrams import GramIndex
from hypothesis import given, strategies as st
from itertools import pairwise


@given(
    st.lists(
        st.integers(0, 2 ** 16 - 1), min_size=1,
    )
)
def test_word_trie(tokens: list[int]):
    # Construct index
    index = GramIndex(tokens)

    # Check unigram counts
    for t in tokens:
        assert index.count([t]) == tokens.count(t)

    # Check bigram counts
    bigrams = list(pairwise(tokens))
    for b in bigrams:
        assert index.count(list(b)) == bigrams.count(b)
