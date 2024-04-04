class InMemoryIndex:
    """An n-gram index."""

    def __init__(self, tokens: list[int], verbose: bool) -> None:
        ...
    
    @staticmethod
    def from_token_file(path: str, verbose: bool, token_limit: int | None) -> "InMemoryIndex":
        """Construct a `InMemoryIndex` from a file containing raw little-endian tokens."""

    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of `query` in the index."""

    def positions(self, query: list[int]) -> list[int]:
        """Returns an unordered list of positions where `query` starts in `text`."""

    def count_next(self, query: list[int], vocab: int | None) -> list[int]:
        """Count the occurrences of each token directly following `query`."""

    def batch_count_next(self, queries: list[list[int]], vocab: int | None) -> list[list[int]]:
        """Count the occurrences of each token that directly follows each sequence in `queries`."""

    def sample(self, query: list[int], n: int, k: int) -> list[int]:
        """Autoregressively sample k characters from conditional distributions based 
        on the previous (n - 1) characters (n-gram prefix) in the sequence. If there are fewer than 
        (n - 1) characters all available characters are used.""" 
    
    def batch_sample(self, query: list[int], n: int, k: int, num_samples: int) -> list[list[int]]:
        """Autoregressively sample num_samples of k characters each from conditional distributions based 
        on the previous (n - 1) characters (n-gram prefix) in the sequence. If there are fewer than 
        (n - 1) characters all available characters are used.""" 

    def is_sorted(self) -> bool:
        """Check if the index's suffix table is sorted lexicographically. 
        This is always true for valid indices."""

class MemmapIndex:
    """An n-gram index backed by a memory-mapped file."""

    def __init__(self, token_file: str, index_file: str) -> None:
        """Load a prebuilt memory-mapped index from a pair of files."""

    @staticmethod
    def build(token_file: str, index_file: str, verbose: bool) -> "MemmapIndex":
        """Build a memory-mapped index from a token file."""
    
    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of `query` in the index."""

    def positions(self, query: list[int]) -> list[int]:
        """Returns an unordered list of positions where `query` starts in `text`."""

    def count_next(self, query: list[int], vocab: int | None) -> list[int]:
        """Count the occurrences of each token directly following `query`."""

    def batch_count_next(self, queries: list[list[int]], vocab: int | None) -> list[list[int]]:
        """Count the occurrences of each token that directly follows each sequence in `queries`."""

    def sample(self, query: list[int], n: int, k: int) -> list[int]:
        """Autoregressively k characters from conditional distributions based 
        on the previous (n - 1) characters (n-gram prefix) in the sequence. If there are fewer than 
        (n - 1) characters all available characters are used.""" 
   
    def batch_sample(self, query: list[int], n: int, k: int, num_samples: int) -> list[list[int]]:
        """Autoregressively samples num_samples of k characters each from conditional distributions based 
        on the previous (n - 1) characters (n-gram prefix) in the sequence. If there are fewer than 
        (n - 1) characters all available characters are used."""
   
    def is_sorted(self) -> bool:
        """Check if the index's suffix table is sorted lexicographically. 
        This is always true for valid indices."""