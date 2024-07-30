from expecttest import assert_eq
from tokengrams import InMemoryIndex
from tokengrams import Sampler


# def test_sample_unsmoothed_exists():
#     tokens = [1, 2, 3, 4]
#     index = InMemoryIndex(tokens, verbose=False)
#     sampler = Sampler.from_in_memory_index(index)
    a = ord("a")

    seqs = sampler.sample_unsmoothed([a], 3, 10, 20)

    assert seqs[0][-1] == a
    assert seqs[19][-1] == a


# def test_sample_unsmoothed_empty_query_exists(self):
#     sampler = Sampler(CountableIndex.from_str("aaa"))
#     seqs = sampler.sample_unsmoothed([], 3, 10, 20)

#     self.assertEqual(seqs[0][-1], "a"[0])
#     self.assertEqual(seqs[19][-1], "a"[0])

# def test_sample_smoothed_exists(self):
#     sampler = Sampler(CountableIndex.from_str("aabbccabccba"))
#     tokens = sampler.sample_smoothed([ord("a")], 3, 10, 1)[0]

#     self.assertEqual(len(tokens), 11)

# def test_sample_smoothed_unigrams_exists(self):
#     sampler = Sampler(CountableIndex.from_str("aabbccabccba"))
#     tokens = sampler.sample_smoothed([ord("a")], 1, 10, 10)[0]

#     self.assertEqual(len(tokens), 11)

# def test_prop_sample(self):
#     def prop(s):
#         if len(s) < 2:
#             return True

#         sampler = Sampler(CountableIndex.from_str(s))
#         query = s[:1]
#         got = sampler.sample_unsmoothed(query, 2, 1, 1)[0]
#         return got[0] in s

#     # Example property-based testing call (assuming usage with a property-based testing library)
#     # qc(prop)  # This line is illustrative. Python's property testing might be done with Hypothesis.

# def test_smoothed_probs_exists(self):
#     index = CountableIndex.from_str("aaa")
#     sampler = Sampler(index)
#     query = [ord("a")]
#     vocab = ord("c") + 1
#     a = ord("a")
#     c = ord("c")

#     smoothed_probs = sampler.smoothed_probs(query, vocab)
#     bigram_counts = index.count_next(query, vocab)
#     unsmoothed_probs = [x / sum(bigram_counts) for x in bigram_counts]

#     self.assertEqual(unsmoothed_probs[a], 0.0)
#     self.assertAlmostEqual(unsmoothed_probs[c], 1.0)
#     self.assertTrue(smoothed_probs[a] > 0.1)
#     self.assertTrue(smoothed_probs[c] < 1.0)

# def test_smoothed_probs_empty_query_exists(self):
#     index = CountableIndex.from_str("aaa")
#     sampler = Sampler(index)

#     probs = sampler.smoothed_probs([], ord("a") + 1)
#     residual = abs(sum(probs) - 1.0)

#     self.assertLess(residual, 1e-4)
