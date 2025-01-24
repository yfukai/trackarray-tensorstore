import pytest
import trackarray_tensorstore as tta
from trackarray_tensorstore._io import FilesPropsIO
import numpy as np
import itertools

def _test_write_read_same(cls, trackarr_from_name, tmp_path, **kwargs):
    ta, labels, splits = trackarr_from_name("original")
    bbox_df = tta._trackarray._bbox_dict_to_df(ta._bboxes_dict)
    termination_annotations = {int(i):f"test_{i}" for i in np.unique(labels)}
    readwrite = cls(tmp_path/"test", tmp_path/"test", **kwargs)
    readwrite.write(bbox_df, splits, termination_annotations)
    bbox_df2, splits2, termination_annotations2 = readwrite.read()
    assert bbox_df.equals(bbox_df2)
    assert splits == splits2
    assert termination_annotations == termination_annotations2


@pytest.mark.parametrize("cls", [FilesPropsIO])
def test_write_read_same(cls, trackarr_from_name, tmp_path):
    
    if cls == FilesPropsIO:
        additional_args = {"dataframe_filetype":
            [dft for dft in FilesPropsIO.DataFileType]}
    
    for arg in itertools.product(*additional_args.values()):
        _test_write_read_same(cls, trackarr_from_name, tmp_path, **dict(zip(additional_args.keys(), arg)))
    
    
    
    