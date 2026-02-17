# src/__init__.py
from importlib import import_module
import sys

# Aggregate core sub‑modules so a caller imports the package once instead of listing each file.
__all__ = [
    "front_dataset_handler",
    "env_options",
    "semantic_search"
]

for _mod in __all__:
    mod = import_module(f".{_mod}", __name__)  
    globals()[_mod] = mod                      
    sys.modules[_mod] = mod # Pre‑register bare names so intra‑package imports (e.g., `import front_dataset_handler`) succeed.