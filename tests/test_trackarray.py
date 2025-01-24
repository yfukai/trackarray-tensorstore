import re
from copy import deepcopy

import numpy as np
import pytest
import tensorstore as ts
from skimage.util import map_array

import trackarray_tensorstore as tta
from ._utils import compare_nested_structures


def test_to_bbox_df(labels_dir):
    labels = np.load(labels_dir / "original_labels.npy")
    bbox_df = tta.to_bbox_df(labels).set_index("frame")
    for frame in range(len(labels)):
        _df = bbox_df.loc[frame]
        for _, row in _df.iterrows():
            label = row["label"]
            ind = np.nonzero(labels[frame] == label)
            assert row["min_x"] == ind[1].min()
            assert row["max_x"] == ind[1].max() + 1
            assert row["min_y"] == ind[0].min()
            assert row["max_y"] == ind[0].max() + 1

def test_df_conversion(labels_dir):
    labels = np.load(labels_dir / "original_labels.npy")
    bbox_df = tta.to_bbox_df(labels)
    bbox_df2 = bbox_df.copy().sort_values(["frame","label"]).reset_index(drop=True)
    # make sure the columns starts by frame and label
    bbox_df2 = bbox_df2[["frame","label"]+list(bbox_df2.columns[2:])]
    test_dict = tta._trackarray._bbox_df_to_dict(bbox_df)
    bbox_df3 = tta._trackarray._bbox_dict_to_df(test_dict)
    bbox_df3 = bbox_df3.sort_values(["frame","label"]).reset_index(drop=True)
    bbox_df3 = bbox_df3[["frame","label"]+list(bbox_df3.columns[2:])]
    assert bbox_df2.equals(bbox_df3)
    

def test_trackarr_fixture_always_new(labels_dir, trackarr_from_name):
    ta, _, _ = trackarr_from_name("original")
    with ts.Transaction() as txn:
        ta.add_mask(0, 1, (10, 10), np.ones((100, 100), dtype=bool), txn)
    ta2, _, _ = trackarr_from_name("original")
    assert np.any(np.array(ta.array) != np.array(ta2.array))
    np.load(labels_dir / "original_labels.npy")
    assert np.all(np.array(ta2.array) == np.load(labels_dir / "original_labels.npy"))


def test_validate(trackarr_from_name, all_label_filenames):
    assert len(all_label_filenames) > 0
    for filename in all_label_filenames:
        ta, _, _ = trackarr_from_name(filename)
        assert ta.is_valid()
        ta._bboxes_dict[1].loc[0, "min_y"] = 0
        assert not ta.is_valid()


def test_delete_mask(trackarr_from_name):
    ta, labels, _ = trackarr_from_name("original")
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    map_dict = {l: l for l in unique_labels}

    for label, grp in ta._bboxes_dict.items():
        for frame in grp.index:
            ta, labels, split_dict = trackarr_from_name("original")
            termination_annotations = deepcopy(ta.termination_annotations)

            with ts.Transaction() as txn:
                ta.delete_mask(frame, label, txn)
            assert ta.is_valid()

            split_dict2 = ta.splits

            map_dict2 = deepcopy(map_dict)
            map_dict2.pop(label)
            labels2 = labels.copy()
            labels2[frame] = map_array(
                labels[frame],
                np.array(list(map_dict2.keys())),
                np.array(list(map_dict2.values())),
            )
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
            ta.add_mask(
                frame, new_label, (10, 10), np.ones((100, 100), dtype=bool), txn
            )
        assert np.all(np.array(ta.array)[frame, 10:110, 10:110] == new_label)
        assert ta.is_valid()


def test_update_mask(trackarr_from_name):
    ta, labels, _ = trackarr_from_name("original")
    assert 1 in labels[0].ravel()
    original_inds = np.where(labels[0] == 1)
    with ts.Transaction() as txn:
        ta.update_mask(0, 1, (10, 10), np.ones((100, 100), dtype=bool), txn)
    assert ta.is_valid()
    assert np.all(np.array(ta.array)[0, 10:110, 10:110] == 1)

    for i in zip(*original_inds):
        if i[0] in range(10, 110) and i[1] in range(10, 110):
            assert np.all(np.array(ta.array)[0, i[0], i[1]] == 1)
        else:
            assert np.all(np.array(ta.array)[0, i[0], i[1]] != 1)


@pytest.mark.parametrize("test_name", ["frame2_8terminate", "frame4_1terminate"])
def test_terminate_track(trackarr_from_name, test_name):
    ta, labels, _ = trackarr_from_name("original")
    ta2, labels2, _ = trackarr_from_name(test_name)
    assert np.any(labels != labels2)
    terminate_frame, terminate_label = re.search(
        r"frame(\d+)_(\d+)terminate", test_name
    ).groups()
    terminate_frame = int(terminate_frame)
    terminate_label = int(terminate_label)

    with ts.Transaction() as txn:
        ta.terminate_track(terminate_frame, terminate_label, "test_annotation", txn)
    assert np.all(np.array(ta.array) == labels2)
    assert compare_nested_structures(ta.splits, ta2.splits)
    ta2.termination_annotations[terminate_label] = "test_annotation"
    assert compare_nested_structures(
        ta.termination_annotations, ta2.termination_annotations
    )


def test_split(trackarr_from_name):
    ta2, labels2, _ = trackarr_from_name("frame7_3split_to_18_and_20")
    ta, labels, _ = trackarr_from_name("original")
    assert np.any(labels != labels2)
    daughter_start_frame = 7
    parent_trackid = 3
    daughter_tracks = [18, 3]
    with ts.Transaction() as txn:
        ta.add_split(daughter_start_frame, parent_trackid, daughter_tracks, txn)
    assert np.all(np.array(ta.array) == labels2)
    assert compare_nested_structures(ta.splits, ta2.splits)


break_test_names = [
    "frame3_8break_change_after_to_5",
    "frame3_8break_change_after_to_20",
    "frame4_8break_change_after_to_5",
    "frame4_8break_change_after_to_20",
    "frame3_8break_change_before_to_5",
    "frame3_8break_change_before_to_20",
    "frame4_8break_change_before_to_5",
    "frame4_8break_change_before_to_20",
    "frame4_11break_change_after_to_20",
    "frame4_11break_change_before_to_20",
    "frame5_11break_change_after_to_20",
    "frame5_11break_change_before_to_20",
]


@pytest.mark.parametrize("test_name", break_test_names)
def test_break_track(trackarr_from_name, test_name):
    ta, labels, splits1 = trackarr_from_name("original")
    ta2, labels2, splits2 = trackarr_from_name(test_name)
    assert np.any(labels != labels2) or not compare_nested_structures(splits1, splits2)
    new_start_frame, divide_label, direction, dest_label = re.search(
        r"frame(\d+)_(\d+)break_change_(.+)_to_(\d+)", test_name
    ).groups()
    new_start_frame = int(new_start_frame)
    divide_label = int(divide_label)
    dest_label = int(dest_label)
    change_after = direction == "after"

    with ts.Transaction() as txn:
        if dest_label == 5 and test_name != "frame4_8break_change_after_to_5":
            with pytest.raises(ValueError):
                ta.break_track(
                    new_start_frame,
                    divide_label,
                    change_after,
                    txn,
                    new_trackid=dest_label,
                )
            return
        else:
            ta.break_track(
                new_start_frame, divide_label, change_after, txn, new_trackid=dest_label
            )
    assert ta.is_valid()
    assert np.all(np.array(ta.array) == labels2)
    assert compare_nested_structures(ta.splits, ta2.splits)
