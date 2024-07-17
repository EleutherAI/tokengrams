from tokengrams import MemmapIndex

class ShardedMemmapIndex:
    def __init__(self, files: list[tuple[str, str]]) -> None:
        """Load a set of shards of a prebuilt memory-mapped sharded index from pairs of files."""
        self.shards = [
            MemmapIndex(token_file, index_file) 
            for token_file, index_file in files
        ]

    
    @staticmethod
    def build(files: list[tuple[str, str]], verbose: bool) -> "ShardedMemmapIndex":
        """Build a memory-mapped index from several token files."""
        for token_file, index_file in files:
            MemmapIndex.build(token_file, index_file, verbose)
        
        return ShardedMemmapIndex(files)


    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
        return any([shard.contains(query) for shard in self.shards])
    

    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of `query` in the index."""
        return sum([shard.count(query) for shard in self.shards])


    def count_next(self, query: list[int], vocab: int | None) -> list[int]:
        """Count the occurrences of each token directly following `query`."""
        counts = [shard.count_next(query, vocab) for shard in self.shards]
        return [sum(counts[i]) for i in range(len(counts))]


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