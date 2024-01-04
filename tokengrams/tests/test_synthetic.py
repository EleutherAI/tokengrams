from tokengrams import WordTrie
from hypothesis import given, strategies as st


@given(
    st.lists(
        st.tuples(st.text(st.characters(whitelist_categories=["L"])), st.integers(0, 2 ** 63)),
        min_size=1,
        unique_by=lambda x: x[0],
    )
)
def test_word_trie(grams: list[tuple[str, int]]):
    # Construct modified Google n-grams format
    body = "\n".join(f"{k}\t{v}" for k, v in grams)
    text = f"{len(grams)}\n{body}"

    # Construct trie
    trie = WordTrie.from_texts([text])
    assert trie.num_grams() == len(grams)

    for k, v in grams:
        assert trie.find(k) == v
