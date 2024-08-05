from typing import List, Optional, Tuple
from tokengrams import MemmapIndexU16, MemmapIndexU32, ShardedMemmapIndexU16, ShardedMemmapIndexU32, InMemoryIndexU16, InMemoryIndexU32

class MemmapIndex:
    def __init__(self, text_path: str, table_path: str, vocab: Optional[int] = None):
        if vocab is None or vocab < 2**16:
            self._index = MemmapIndexU16(text_path, table_path)
        else:
            self._index = MemmapIndexU32(text_path, table_path)

    def __getattr__(self, name):
        return getattr(self._index, name)

    @classmethod
    def build(cls, text_path: str, table_path: str, verbose: bool = False, vocab: Optional[int] = None) -> 'MemmapIndex':
        if vocab is None or vocab < 2**16:
            index = MemmapIndexU16.build(text_path, table_path, verbose)
        else:
            index = MemmapIndexU32.build(text_path, table_path, verbose)
        instance = cls.__new__(cls)
        instance._index = index
        return instance

class ShardedMemmapIndex:
    def __init__(self, files: List[Tuple[str, str]], vocab: Optional[int] = None):
        if vocab is None or vocab < 2**16:
            self._index = ShardedMemmapIndexU16(files)
        else:
            self._index = ShardedMemmapIndexU32(files)

    def __getattr__(self, name):
        return getattr(self._index, name)

    @classmethod
    def build(cls, paths: List[Tuple[str, str]], verbose: bool = False, vocab: Optional[int] = None) -> 'ShardedMemmapIndex':
        if vocab is None or vocab < 2**16:
            index = ShardedMemmapIndexU16.build(paths, verbose)
        else:
            index = ShardedMemmapIndexU32.build(paths, verbose)
        instance = cls.__new__(cls)
        instance._index = index
        return instance

class InMemoryIndex:
    def __init__(self, tokens: List[int], verbose: bool = False, vocab: Optional[int] = None):
        if vocab is None or vocab <= 65535:  # 65535 is u16::MAX
            self._index = InMemoryIndexU16(tokens, verbose)
        else:
            self._index = InMemoryIndexU32(tokens, verbose)

    def __getattr__(self, name):
        return getattr(self._index, name)

    @classmethod
    def from_pretrained(cls, path: str, verbose: bool) -> 'InMemoryIndex':
        # We need to determine the type based on the saved file
        # This is a simplification; you might need to implement a way to store the type information
        try:
            index = InMemoryIndexU16.from_token_file(path, verbose, None)
        except:
            index = InMemoryIndexU32.from_token_file(path, verbose, None)
        instance = cls.__new__(cls)
        instance._index = index
        return instance

    @classmethod
    def from_token_file(cls, path: str, verbose: bool = False, token_limit: Optional[int] = None, vocab: Optional[int] = None) -> 'InMemoryIndex':
        if vocab is None or vocab <= 65535:
            index = InMemoryIndexU16.from_token_file(path, verbose, token_limit)
        else:
            index = InMemoryIndexU32.from_token_file(path, verbose, token_limit)
        instance = cls.__new__(cls)
        instance._index = index
        return instance