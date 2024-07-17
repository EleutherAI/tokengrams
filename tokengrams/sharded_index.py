from tokengrams import MemmapIndex
import random

class ShardedMemmapIndex:
    def __init__(self, files: list[tuple[str, str]]) -> None:
        """Load a set of shards of a prebuilt memory-mapped sharded index from pairs of files."""
        self.shards = [
            MemmapIndex(token_file, index_file) 
            for token_file, index_file in files
        ]
        assert all([shard.is_sorted() for shard in self.shards])

    
    @staticmethod
    def build(files: list[tuple[str, str]], verbose: bool) -> "ShardedMemmapIndex":
        """Build a memory-mapped index from several token files."""
        for token_file, index_file in files:
            MemmapIndex.build(token_file, index_file, verbose)
        
        return ShardedMemmapIndex(files)


    def is_sorted(self) -> bool:
        """Check if each individual suffix table shard is sorted lexicographically. 
        This is always true for valid indices."""
        self.shards.iter().all(|shard| shard.is_sorted())
    
    
    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
        return any([shard.contains(query) for shard in self.shards])
    

    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of `query` in the index."""
        return sum([shard.count(query) for shard in self.shards])


    def count_next(self, query: list[int], vocab: int | None) -> list[int]:
        """Count the occurrences of each token directly following `query`."""
        counts = [shard.count_next(query, vocab) for shard in self.shards]
        return [sum([count[i] for count in counts]) for i in range(len(counts[0]))]


    def batch_count_next(self, queries: list[list[int]], vocab: int | None) -> list[list[int]]:
        """Count the occurrences of each token that directly follows each sequence in `queries`."""
        vocab = vocab or 2**16 + 1

        shard_counts = [shard.batch_count_next(queries, vocab) for shard in self.shards]
        counts = [[0] * vocab] * len(queries)
        for shard_count in shard_counts:
            for i, query_count in enumerate(shard_count):
                for j, count in enumerate(query_count):
                    counts[i][j] += count

        return counts


    def sample(self, query: list[int], n: int, k: int, num_samples: int, vocab: int | None) -> list[list[int]]:
        """Autoregressively k characters from conditional distributions based 
        on the previous (n - 1) characters (n-gram prefix) in the sequence. If there are fewer than 
        (n - 1) characters all available characters are used.""" 
        sequences = [query.copy()] * num_samples
        for sequence in sequences:
            for _ in range(k):
                # look at the previous (n - 1) characters to predict the n-gram completion
                start = max(0, len(sequence) - (n - 1))
                prev = sequence[start:]

                counts = self.count_next(prev, vocab)
                try:
                    sampled_index = random.choices(
                        range(len(counts)),
                        weights=counts, k=1
                    )[0]
                except ValueError:
                    # Handle edge case where final token in corpus only occurs once 
                    # and has no continuations by falling back to unigram distribution
                    counts = self.count_next([], vocab)
                    sampled_index = random.choices(
                        range(len(counts)),
                        weights=counts, k=1
                    )[0]
                sequence.append(sampled_index)

        return sequences