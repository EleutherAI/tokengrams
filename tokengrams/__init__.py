import os
print(os.getcwd())

from .tokengrams import (
    InMemoryIndex,
    MemmapIndex,
)