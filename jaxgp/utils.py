from typing import List, Dict


def concat_dictionaries(*args: List[Dict]) -> Dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the key-value pair of the last supplied
    dictionary will be used.
    """
    result = {}
    for d in args:
        result.update(d)
    return result
