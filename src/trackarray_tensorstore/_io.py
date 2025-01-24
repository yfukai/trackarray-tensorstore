import json
from enum import Enum
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
from ._trackarray import TrackArray

def read_files(ts_array, bboxes_df_file_path, props_json_file_path = None):
    writer = FilesPropsIO(bboxes_df_file_path, props_json_file_path)
    return TrackArray(ts_array,property_writer=writer)

class FilesPropsIO:
    class DataFileType(Enum):
        CSV = 1
        FEATHER = 2
        PARQUET = 3
        HDF5 = 4
        
    EXTENSION_MAP = {
        DataFileType.CSV: ".csv",
        DataFileType.FEATHER: ".feather",
        DataFileType.PARQUET: ".parquet",
        DataFileType.HDF5: ".h5",
    }
        
    def __init__(
        self,
        bboxes_df_file_path,
        props_json_file_path = None,
        dataframe_filetype=DataFileType.CSV,
    ):
        bboxes_df_file_path = Path(bboxes_df_file_path)
        ext = FilesPropsIO.EXTENSION_MAP[dataframe_filetype]
        if props_json_file_path is None:
            props_json_file_path = bboxes_df_file_path.with_suffix(".json")
        elif props_json_file_path.suffix == "" or \
            props_json_file_path.suffix != ".json":
            props_json_file_path = props_json_file_path.with_suffix(".json")
        if bboxes_df_file_path.suffix == "" or \
            bboxes_df_file_path.suffix != ext:
            bboxes_df_file_path = bboxes_df_file_path.with_suffix(ext)

        self.bboxes_df_file_path = bboxes_df_file_path
        self.props_json_file_path = props_json_file_path
        self.dataframe_filetype = dataframe_filetype

    def read(self) -> Tuple[Dict, Dict, pd.DataFrame]:
        if self.dataframe_filetype == FilesPropsIO.DataFileType.CSV:
            df = pd.read_csv(self.bboxes_df_file_path, index_col=0)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.FEATHER:
            df = pd.read_feather(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.PARQUET:
            df = pd.read_parquet(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.HDF5:
            df = pd.read_hdf(self.bboxes_df_file_path,key="bbox_df")
        with open(self.props_json_file_path, "r") as file:
            props = json.load(file)
        splits = {int(k):list(map(int,vs)) 
                           for k,vs in props["splits"].items()}
        termination_annotations = {int(k):str(v) 
                    for k,v in props["termination_annotations"].items()}
        return df, splits, termination_annotations, props["attrs"]

    def write(self, df: pd.DataFrame, splits: Dict, termination_annotations: Dict, attrs: Dict):
        if self.dataframe_filetype == FilesPropsIO.DataFileType.CSV:
            df.to_csv(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.FEATHER:
            df.to_feather(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.PARQUET:
            df.to_parquet(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.HDF5:
            df.to_hdf(self.bboxes_df_file_path,key="bbox_df")
        with open(self.props_json_file_path, "w") as file:
            json.dump(
                {"splits": splits, "termination_annotations": termination_annotations,"attrs":attrs},
                file,
            )
