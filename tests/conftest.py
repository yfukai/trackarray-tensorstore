import json
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

import tensorstore_trackarr as tta
from ._utils import get_read_spec, get_write_spec



@pytest.fixture
def labels_dir(shared_datadir):
    return Path(shared_datadir) / "test_labels"


@pytest.fixture
def all_label_filenames(labels_dir):
    return [
        Path(f).stem.replace("_labels", "") for f in glob(str(labels_dir / "*.npy"))
    ]


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