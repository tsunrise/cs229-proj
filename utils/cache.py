import os
from typing import Callable
import pickle

CACHE_PATH = ".cs229_cache"

def cached(object_fn: Callable, file_name: str):
    """Decorator to cache the result of a function call.

    Args:
        object: The object to cache.
        path: The path to the cache file.

    Returns:
        The cached object.
    """
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    cache_path = os.path.join(CACHE_PATH, file_name)
    if os.path.exists(cache_path):
        print("Loading cached object from {}".format(cache_path))
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        with open(cache_path, "wb") as f:
            obj = object_fn()
            pickle.dump(obj, f)
        print("Cached object to {}".format(cache_path))
        return obj