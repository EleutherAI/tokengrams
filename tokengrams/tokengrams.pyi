class GramIndex:
    """An n-gram index."""

    def __init__(self, tokens: list[int]) -> None:
        ...
    
    @staticmethod
    def from_token_file(path: str, token_limit: int | None) -> "GramIndex":
        """Construct a `GramIndex` from a file containing raw little-endian tokens."""

    def contains(self, query: list[int]) -> bool:
        """Check if `query` has nonzero count. Faster than `count(query) > 0`."""
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of a query in the index."""
