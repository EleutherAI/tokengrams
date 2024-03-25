class InMemoryIndex:
    """An n-gram index."""

    def __init__(self, tokens: list[int]) -> None:
        ...
    
    @staticmethod
    def from_token_file(path: str, token_limit: int | None) -> "InMemoryIndex":
        """Construct a `InMemoryIndex` from a file containing raw little-endian tokens."""

    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of a query in the index."""

    def sample(self, query: list[int]) -> int:
        """Sample a character with a probability corresponding to the number of occurrences directly after the query."""    
    


class MemmapIndex:
    """An n-gram index backed by a memory-mapped file."""

    def __init__(self, token_file: str, index_file: str) -> None:
        """Load a prebuilt memory-mapped index from a pair of files."""

    @staticmethod
    def build(token_file: str, index_file: str) -> "MemmapIndex":
        """Build a memory-mapped index from a token file."""
    
    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of a query in the index."""

    def sample_ngrams(self, query: list[int], n: int, k: int, batch: int = 1) -> int:
        """Sample k characters with probability corresponding to the number of occurrences directly after the previous n characters."""    
   