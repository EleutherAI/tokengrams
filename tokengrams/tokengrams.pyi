# TODO: Flesh this out
class GramIndex:
    """An n-gram index."""

    def __init__(self, tokens: list[int]) -> None:
        ...
    
    def count(self, query: list[int]) -> int:
        """Count the number of occurrences of a query in the index."""

    def search(self, query: list[int]) -> list[int]:
        """Search for the positions of a query in the index."""
