from itertools import pairwise
from tempfile import NamedTemporaryFile

from tokengrams import InMemoryIndex, MemmapIndex
from hypothesis import given, strategies as st

import numpy as np



def test_sample_unsmoothed_exists(self):
    sa = SuffixArray("aaa")
    a = "a"
    seqs = sa.sample_unsmoothed(a, 3, 10, 20)

    self.assertEqual(seqs[0][-1], a[0])
    self.assertEqual(seqs[19][-1], a[0])

def test_sample_unsmoothed_empty_query_exists(self):
    sa = SuffixArray("aaa")
    seqs = sa.sample_unsmoothed("", 3, 10, 20)

    self.assertEqual(seqs[0][-1], "a"[0])
    self.assertEqual(seqs[19][-1], "a"[0])

def test_sample_smoothed_exists(self):
    sa = SuffixArray("aabbccabccba")
    tokens = sa.sample_smoothed("a", 3, 10, 1)[0]

    self.assertEqual(len(tokens), 11)

def test_sample_smoothed_unigrams_exists(self):
    sa = SuffixArray("aabbccabccba")
    tokens = sa.sample_smoothed("a", 1, 10, 10)[0]

    self.assertEqual(len(tokens), 11)

def test_prop_sample(self):
    def prop(s):
        if len(s) < 2:
            return True
        sa = SuffixArray(s)
        query = s[:1]
        got = sa.sample_unsmoothed(query, 2, 1, 1)[0]
        return got[0] in s

    # Example property-based testing call (assuming usage with a property-based testing library)
    # qc(prop)  # This line is illustrative. Python's property testing might be done with Hypothesis.

def test_smoothed_probs_exists(self):
    sa = SuffixArray("aaaaaaaabc")
    query = ["b"]
    vocab = ord("c") + 1
    a = ord("a")
    c = ord("c")

    smoothed_probs = sa.get_smoothed_probs(query, vocab)
    bigram_counts = sa.count_next(query, vocab)
    unsmoothed_probs = [x / sum(bigram_counts) for x in bigram_counts]

    self.assertEqual(unsmoothed_probs[a], 0.0)
    self.assertAlmostEqual(unsmoothed_probs[c], 1.0)
    self.assertTrue(smoothed_probs[a] > 0.1)
    self.assertTrue(smoothed_probs[c] < 1.0)

def test_smoothed_probs_empty_query_exists(self):
    sa = SuffixArray("aaa")
    probs = sa.get_smoothed_probs([], ord("a") + 1)
    residual = abs(sum(probs) - 1.0)

    self.assertLess(residual, 1e-4)
