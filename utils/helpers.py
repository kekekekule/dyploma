from collections import defaultdict
from functools import reduce


def _defaultdict_recursive():
    return defaultdict(_defaultdict_recursive)


def _cast_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: _cast_to_dict(v) for k, v in d.items()}
    return d


def unflatten_dict(instance: dict, sep: str = "."):
    result = defaultdict(_defaultdict_recursive)

    for complex_key, value in instance.items():
        keys_path = complex_key.split(sep)
        last_key_reference = result
        for key in keys_path[:-1]:
            last_key_reference = last_key_reference.__getitem__(key)
        last_key_reference.__setitem__(keys_path[-1], value)

    return _cast_to_dict(result)


def merge_dicts_recursive(first_dict: dict, second_dict: dict):
    first_keyset = set(first_dict.keys())
    second_keyset = set(second_dict.keys())

    return {
        **{
            key: merge_dicts_recursive(first_dict[key], second_dict[key])
            for key in first_keyset.intersection(second_keyset)
        },
        **{key: first_dict[key] for key in first_keyset.difference(second_keyset)},
        **{key: second_dict[key] for key in second_keyset.difference(first_keyset)},
    }
