import tensorstore_trackarr as tta
import numpy as np
import pytest
import json
from pathlib import Path
import tensorstore as ts
from glob import glob
from copy import deepcopy
from skimage.util import map_array
import itertools
import re

def get_spec(ndims):
    return {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": None
        },
    "metadata": {
        "shape": None,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": ([1]*(ndims-2))+[32768, 32768]}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": [{"name" : "sharding_indexed",
                    "configuration":{
                    "chunk_shape": ([1]*(ndims-2))+[512, 512],
                    "codecs":[{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}],
                    }}],
        "data_type": "uint32",
    },
    'context': {
            'cache_pool': {
                'total_bytes_limit': 100_000_000
            }
        },
        'recheck_cached_data':'open',
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
        return all(compare_nested_structures(el1, el2) for el1, el2 in zip(data1, data2))
    else:
        # For other types, compare directly
        return data1 == data2

@pytest.fixture
def labels_dir(shared_datadir):
    return Path(shared_datadir) / "test_labels"

@pytest.fixture
def all_label_filenames(labels_dir):
    return [Path(f).stem.replace("_labels","") for f in glob(str(labels_dir / "*.npy"))]

@pytest.fixture
def trackarr(tmpdir):
    counter = 0
    
    def _generate_trackarr(labels, split_dict):
        nonlocal counter
        counter += 1
        zarr_filename = Path(tmpdir) / (f"test{counter}.zarr")
        _write_spec = get_write_spec(zarr_filename, labels.shape)
        ts.open(_write_spec).result().write(labels).result()
    
        _read_spec = get_read_spec(zarr_filename)
        ts_read = ts.open(_read_spec).result()
        bbox_df = tta.to_bbox_df(labels)
        unique_labels = np.unique(bbox_df.index.get_level_values("label"))
        unique_labels = unique_labels[unique_labels != 0]
        
        termination_annotations = {
            l: f"terminate {l}" for l in unique_labels if l % 2 == 0
        }
        
        return tta.TrackArray(ts_read, split_dict, termination_annotations, bbox_df)
    
    return _generate_trackarr

def convert_keys_and_values_to_int(d):
    """
    Converts all keys and values of a dictionary to integers,
    assuming they are convertible.

    Args:
        d: A dictionary to process.

    Returns:
        A new dictionary with keys and values as integers.
    """
    return {int(key): [int(v) for v in value] for key, value in d.items()}

@pytest.fixture
def trackarr_from_name(labels_dir, trackarr):
    def _trackarr_from_name(filename):
        labels = np.load(labels_dir / (filename + "_labels.npy"))
        with open(labels_dir / (filename + "_split_dict.json")) as f:
            split_dict = json.load(f, object_hook=convert_keys_and_values_to_int)
        return trackarr(labels, split_dict), labels, split_dict
    return _trackarr_from_name

def test_to_bbox_df(labels_dir):
    labels = np.load(labels_dir / 'original_labels.npy')
    bbox_df = tta.to_bbox_df(labels)
    for frame in range(len(labels)):
        _df = bbox_df.loc[frame]
        for label, row in _df.iterrows():
            ind = np.nonzero(labels[frame] == label)
            assert row['min_x'] == ind[1].min()
            assert row['max_x'] == ind[1].max()+1
            assert row['min_y'] == ind[0].min()
            assert row['max_y'] == ind[0].max()+1
            
def test_update_track_df():
    pass

def test_trackarr_fixture_always_new(labels_dir, trackarr_from_name):
    ta, _, _ = trackarr_from_name("original")
    with ts.Transaction() as txn:
        ta.swap_tracks(1,2,txn)
    ta2, _, _ = trackarr_from_name("original")
    assert np.any(np.array(ta.array) != np.array(ta2.array))
    np.load(labels_dir / "original_labels.npy")
    assert np.all(np.array(ta2.array) == np.load(labels_dir / "original_labels.npy"))

def test_validate(trackarr_from_name, all_label_filenames):
    assert len(all_label_filenames) > 0
    for filename in all_label_filenames:
        ta,_,_ = trackarr_from_name(filename)
        assert ta.is_valid()
        ta.bboxes_df.loc[(0,1), "min_y"] = 0
        assert not ta.is_valid()

def test_swap_tracks(trackarr_from_name):
    _, labels, _ = trackarr_from_name("original")
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    map_dict = {l:l for l in unique_labels}
    
    for l1, l2 in itertools.combinations(unique_labels, 2):
        ta, labels, split_dict = trackarr_from_name("original")
        termination_annotations = deepcopy(ta.termination_annotations)
        
        with ts.Transaction() as txn:
            ta.swap_tracks(l1, l2, txn)
        split_dict2 = ta.splits
        
        map_dict2 = deepcopy(map_dict)
        map_dict2[l1] = l2
        map_dict2[l2] = l1
        from_vals = np.array(list(map_dict2.keys()))
        to_vals = np.array(list(map_dict2.values()))
        labels2 = map_array(labels, from_vals, to_vals)
        
        assert np.all(labels2 == np.array(ta.array))
        
        split_dict_swapped = swap_values(split_dict, l1, l2)
        assert compare_nested_structures(split_dict_swapped, split_dict2)
        
        termination_annotations_swapped = {}
        for k, v in termination_annotations.items():
            if k == l1:
                termination_annotations_swapped[l2] = v
            elif k == l2:
                termination_annotations_swapped[l1] = v
            else:
                termination_annotations_swapped[k] = v
        assert termination_annotations_swapped == ta.termination_annotations

def test_delete_mask(trackarr_from_name):
    ta, labels, _ = trackarr_from_name("original")    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    map_dict = {l:l for l in unique_labels}
   
    for frame, label in ta.bboxes_df.index: 
        ta, labels, split_dict = trackarr_from_name("original")
        termination_annotations = deepcopy(ta.termination_annotations)
        
        with ts.Transaction() as txn:
            ta.delete_mask(frame, label, txn)
        assert ta.is_valid()

        split_dict2 = ta.splits

        map_dict2 = deepcopy(map_dict)
        map_dict2.pop(label)
        labels2 = labels.copy()
        labels2[frame] = map_array(labels[frame], 
                                   np.array(list(map_dict2.keys())), 
                                   np.array(list(map_dict2.values())))
        assert np.all(labels2 == np.array(ta.array))
        
        if label not in labels2.ravel():
            assert label not in split_dict2
            assert all(label not in v for v in split_dict2.values())
            assert label not in ta.termination_annotations
            
def test_add_mask(trackarr_from_name):
    ta, labels, _ = trackarr_from_name("original")    
    new_label = np.max(labels) + 1
    for frame in range(len(labels)):
        ta, labels, _ = trackarr_from_name("original")    
        with ts.Transaction() as txn:
            ta.add_mask(frame, new_label, (10,10), np.ones((100,100),dtype=bool), txn)
        assert np.all(np.array(ta.array)[frame, 10:110, 10:110] == new_label)
        assert ta.is_valid()
    
def test_update_mask(trackarr_from_name):
    ta, labels, _ = trackarr_from_name("original")
    assert 1 in labels[0].ravel()
    original_inds = np.where(labels[0] == 1)
    with ts.Transaction() as txn:
        ta.update_mask(0, 1, (10,10), np.ones((100,100),dtype=bool), txn)
    assert ta.is_valid()
    assert np.all(np.array(ta.array)[0, 10:110, 10:110] == 1)
    
    for i in zip(*original_inds):
        if i[0] in range(10,110) and i[1] in range(10,110):
            assert np.all(np.array(ta.array)[0, i[0], i[1]] == 1)
        else:
            assert np.all(np.array(ta.array)[0, i[0], i[1]] != 1)

def test_terminate_track(trackarr_from_name):
    test_names = ["frame2_8terminate", "frame4_1terminate"]
    
    for test_name in test_names:
        ta2, labels2, _ = trackarr_from_name(test_name)
        terminate_frame, terminate_label = re.search(r"frame(\d+)_(\d+)terminate", test_name).groups()
        terminate_frame = int(terminate_frame)
        terminate_label = int(terminate_label)

        with ts.Transaction() as txn:
            ta, labels, _ = trackarr_from_name("original")
            assert np.any(labels != labels2)
            ta.terminate_track(terminate_frame, terminate_label, "test_annotation", txn)
        assert np.all(np.array(ta.array) == labels2) 
        assert compare_nested_structures(ta.splits, ta2.splits)
        ta2.termination_annotations[terminate_label] = "test_annotation"
        assert compare_nested_structures(ta.termination_annotations, ta2.termination_annotations)

def test_split(trackarr_from_name):
    ta2, labels2, _ = trackarr_from_name("frame7_3split_to_18_and_20")
    ta, labels, _ = trackarr_from_name("original")
    assert np.any(labels != labels2)
    daughter_start_frame = 7
    parent_trackid = 3
    daughter_tracks = [18, 20]
    with ts.Transaction() as txn:
        ta.add_split(daughter_start_frame, parent_trackid, daughter_tracks, txn)
    assert np.all(np.array(ta.array) == labels2)
    assert compare_nested_structures(ta.splits, ta2.splits)