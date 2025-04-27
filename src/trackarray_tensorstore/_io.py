import json
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import tensorstore as ts

from ._trackarray import TrackArray


def read_files(
    ts_array: ts.TensorStore,
    bboxes_df_file_path: Union[str, Path],
    props_json_file_path: Optional[Union[str, Path]] = None,
) -> TrackArray:
    """
    Read bounding boxes DataFrame and properties from files and return a
    TrackArray instance.

    Creates a FilesPropsIO object using the provided file paths and uses it as
    the property_writer to instantiate a TrackArray from the given tensorstore
    array.

    Parameters
    ----------
    ts_array : ts.TensorStore
        The tensorstore array containing tracking data.
    bboxes_df_file_path : Union[str, Path]
        File path to the bounding boxes DataFrame.
    props_json_file_path : Optional[Union[str, Path]], optional
        File path to the properties JSON file. If not provided, it is inferred
        from bboxes_df_file_path.

    Returns
    -------
    TrackArray
        An instance of TrackArray initialized with the FilesPropsIO as its
        property_writer.
    """
    writer = FilesPropsIO(bboxes_df_file_path, props_json_file_path)
    return TrackArray(ts_array, property_writer=writer)


class FilesPropsIO:
    """A class for reading and writing properties associated with a TrackArray.

    This class provides functionality to read and write a bounding boxes DataFrame
    and corresponding properties (splits, termination annotations, and attributes)
    from designated file paths. The DataFrame can be stored in CSV, Feather,
    Parquet, or HDF5 format.
    """

    class DataFileType(Enum):
        """
        Enumeration of supported file types for bounding box data.
        """

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
        bboxes_df_file_path: Union[str, Path],
        props_json_file_path: Optional[Union[str, Path]] = None,
        dataframe_filetype: "Optional[FilesPropsIO.DataFileType]" = None,
    ) -> None:
        """
        Initialize a FilesPropsIO instance.

        Parameters
        ----------
        bboxes_df_file_path : Union[str, Path]
            File path to the bounding boxes DataFrame.
        props_json_file_path : Optional[Union[str, Path]], optional
            File path to the properties JSON file. If not provided, it is inferred
            by replacing the extension of bboxes_df_file_path with .json.
        dataframe_filetype : FilesPropsIO.DataFileType, optional
            The file type of the bounding boxes DataFrame. 
            If not provided, it is inferred
            from the file extension. The default is None.
        """
        bboxes_df_file_path = Path(bboxes_df_file_path)
        if dataframe_filetype is None:
            if bboxes_df_file_path.suffix in FilesPropsIO.EXTENSION_MAP.values():
                ext = bboxes_df_file_path.suffix
            else:
                raise ValueError(
                    f"Unsupported file extension: {bboxes_df_file_path.suffix}"
                )
        else:
            ext = FilesPropsIO.EXTENSION_MAP[dataframe_filetype]
        if props_json_file_path is None:
            props_json_file_path = bboxes_df_file_path.with_suffix(".json")
        elif isinstance(props_json_file_path, (str, Path)):
            props_json_file_path = Path(props_json_file_path)
            if (
                props_json_file_path.suffix == ""
                or props_json_file_path.suffix != ".json"
            ):
                props_json_file_path = props_json_file_path.with_suffix(".json")
        else:
            raise ValueError("props_json_file_path must be a string or Path object")
        if bboxes_df_file_path.suffix == "" or bboxes_df_file_path.suffix != ext:
            bboxes_df_file_path = bboxes_df_file_path.with_suffix(ext)

        self.bboxes_df_file_path = bboxes_df_file_path
        self.props_json_file_path = props_json_file_path
        self.dataframe_filetype = dataframe_filetype

    def read(self) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, str], Dict]:
        """Read the bounding boxes DataFrame and properties from the designated files.

        Depending on the specified dataframe_filetype, reads the DataFrame in the
        appropriate format. Also reads a JSON file containing splits, termination
        annotations, and additional attributes.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, str], Dict]
            A tuple containing:
            - df : pd.DataFrame
                The bounding boxes DataFrame.
            - splits : dict
                A dictionary mapping track IDs to lists of daughter track IDs.
            - termination_annotations : dict
                A dictionary mapping track IDs to termination annotations.
            - attrs : dict
                Additional attributes read from the JSON file.
        """
        if self.dataframe_filetype == FilesPropsIO.DataFileType.CSV:
            df = pd.read_csv(self.bboxes_df_file_path, index_col=0)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.FEATHER:
            df = pd.read_feather(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.PARQUET:
            df = pd.read_parquet(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.HDF5:
            df = pd.read_hdf(self.bboxes_df_file_path, key="bbox_df")
        with open(self.props_json_file_path) as file:
            props = json.load(file)
        splits = {int(k): list(map(int, vs)) for k, vs in props["splits"].items()}
        termination_annotations = {
            int(k): str(v) for k, v in props["termination_annotations"].items()
        }
        return df, splits, termination_annotations, props["attrs"]

    def write(
        self,
        df: pd.DataFrame,
        splits: Dict[int, List[int]],
        termination_annotations: Dict[int, str],
        attrs: Dict,
    ) -> None:
        """
        Write the bounding boxes DataFrame and properties to the designated files.

        Writes the DataFrame in the format specified by dataframe_filetype and dumps
        the splits, termination annotations, and attributes to a JSON file.

        Parameters
        ----------
        df : pd.DataFrame
            The bounding boxes DataFrame to write.
        splits : Dict[int, List[int]]
            A dictionary mapping track IDs to lists of daughter track IDs.
        termination_annotations : Dict[int, str]
            A dictionary mapping track IDs to termination annotations.
        attrs : Dict
            Additional attributes to write.
        """
        if self.dataframe_filetype == FilesPropsIO.DataFileType.CSV:
            df.to_csv(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.FEATHER:
            df.to_feather(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.PARQUET:
            df.to_parquet(self.bboxes_df_file_path)
        elif self.dataframe_filetype == FilesPropsIO.DataFileType.HDF5:
            df.to_hdf(self.bboxes_df_file_path, key="bbox_df")
        with open(self.props_json_file_path, "w") as file:
            json.dump(
                {
                    "splits": splits,
                    "termination_annotations": termination_annotations,
                    "attrs": attrs,
                },
                file,
            )
