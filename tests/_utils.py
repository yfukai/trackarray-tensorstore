from copy import deepcopy

def get_spec(ndims):
    return {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": None},
        "metadata": {
            "shape": None,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": ([1] * (ndims - 2)) + [32768, 32768]},
            },
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": ([1] * (ndims - 2)) + [512, 512],
                        "codecs": [
                            {
                                "name": "blosc",
                                "configuration": {"cname": "lz4", "clevel": 5},
                            }
                        ],
                    },
                }
            ],
            "data_type": "uint32",
        },
        "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
        "recheck_cached_data": "open",
    }


def get_read_spec(filename):
    read_spec = deepcopy(get_spec(3))
    read_spec["kvstore"]["path"] = str(filename)
    del read_spec["metadata"]["shape"]
    return read_spec


def get_write_spec(filename, shape):
    write_spec = deepcopy(get_spec(3))
    write_spec["create"] = True
    write_spec["delete_existing"] = True
    write_spec["kvstore"]["path"] = str(filename)
    write_spec["metadata"]["shape"] = list(shape)
    return write_spec


def swap_values(data, val1, val2):
    """
    Recursively traverses a nested structure of dicts and lists,
    and swaps all occurrences of val1 and val2.

    Args:
        data: The nested data (dict, list, or other types).
        val1: The first integer to swap.
        val2: The second integer to swap.

    Returns:
        The modified data with swapped values.
    """
    if isinstance(data, dict):
        # If it's a dictionary, recursively apply the function to its values
        return {key: swap_values(value, val1, val2) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, recursively apply the function to its elements
        return [swap_values(element, val1, val2) for element in data]
    elif isinstance(data, int):
        # If it's an integer, check if it matches val1 or val2 and swap
        if data == val1:
            return val2
        elif data == val2:
            return val1
        else:
            return data
    else:
        # For other types, return as-is
        return data


def compare_nested_structures(data1, data2):
    """
    Recursively compares two nested structures (dicts, lists, etc.)
    to check if they are identical.

    Args:
        data1: The first nested structure.
        data2: The second nested structure.

    Returns:
        True if the structures are identical, False otherwise.
    """
    if isinstance(data1, dict) and isinstance(data2, dict):
        # Compare keys and recursively compare values
        if data1.keys() != data2.keys():
            return False
        return all(compare_nested_structures(data1[key], data2[key]) for key in data1)
    elif isinstance(data1, list) and isinstance(data2, list):
        # Compare lists element by element
        if len(data1) != len(data2):
            return False
        return all(
            compare_nested_structures(el1, el2) for el1, el2 in zip(data1, data2)
        )
    else:
        # For other types, compare directly
        return data1 == data2