from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
import tensorstore as ts
from numpy import typing as npt
from skimage.measure import regionprops_table


def to_bbox_df(label: npt.ArrayLike) -> pd.DataFrame:
    """Convert a label array into a bounding box DataFrame.
    The function extracts bounding box properties from each frame of the label array
    using regionprops_table and concatenates the results into a single DataFrame.
    It then renames the bounding box columns to min_y, min_x, max_y, and max_x.

    Parameters
    ----------
    label : npt.ArrayLike
        An array-like object (typically a numpy array) with shape (n_frames, ...)
        where each frame contains labeled regions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the bounding box information for each frame.
    """
    bbox_df = pd.concat(
        [
            pd.DataFrame(
                regionprops_table(np.array(label[frame]), properties=("label", "bbox"))
            ).assign(frame=frame)
            for frame in range(label.shape[0])
        ]
    )
    # bboxtuple
    # Bounding box (min_row, min_col, max_row, max_col).
    # Pixels belonging to the bounding box are in the half-open
    # interval [min_row; max_row) and [min_col; max_col).
    if bbox_df.empty:
        return bbox_df
    bbox_df["min_y"] = bbox_df["bbox-0"]
    bbox_df["min_x"] = bbox_df["bbox-1"]
    bbox_df["max_y"] = bbox_df["bbox-2"]
    bbox_df["max_x"] = bbox_df["bbox-3"]
    del bbox_df["bbox-0"], bbox_df["bbox-1"], bbox_df["bbox-2"], bbox_df["bbox-3"]
    return bbox_df


def _bbox_df_to_dict(bboxes_df):
    """Convert a bounding box DataFrame into a dictionary grouped by label.

    Parameters
    ----------
    bboxes_df : pd.DataFrame
        DataFrame containing bounding box information with a 'label' column.

    Returns
    -------
    dict
        A dictionary mapping each label to its corresponding DataFrame subset,
        with rows indexed by frame.
    """
    return {
        label: grp.set_index("frame").sort_index()
        for label, grp in bboxes_df.groupby("label")
    }


def _bbox_dict_to_df(bboxes_dict):
    """Convert a dictionary of bounding box DataFrames into
       a single concatenated DataFrame.

    Parameters
    ----------
    bboxes_dict : dict
        Dictionary mapping labels to corresponding DataFrame subsets.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame of all bounding box entries with reset index.
    """
    return pd.concat(
        [grp.assign(label=label).reset_index() for label, grp in bboxes_dict.items()]
    ).reset_index(drop=True)


class TrackArray:
    def __init__(
        self,
        ts_array: ts.TensorStore,
        splits: Optional[Dict[int, List[int]]] = None,
        termination_annotations: Optional[Dict[int, str]] = None,
        bboxes_df=None,
        *,
        property_writer=None,
        attrs=None,
    ):
        """
        Initialize a TrackArray instance.

        Parameters
        ----------
        ts_array : ts.TensorStore
            The underlying tensorstore array holding tracking data.
        splits : Optional[Dict[int, List[int]]], optional
            A dictionary specifying track splits, where keys are parent track IDs
            and values are lists of daughter track IDs.
        termination_annotations : Optional[Dict[int, str]], optional
            A dictionary mapping track IDs to termination annotations.
        bboxes_df : pd.DataFrame, optional
            A precomputed DataFrame of bounding boxes.
        property_writer : object, optional
            An object with a 'read' method to load bounding box data and other properties.
        attrs : any, optional
            Additional attributes associated with the track array.

        Raises
        ------
        ValueError
            If neither property_writer is provided nor both splits and termination_annotations are set.
        """
        # FIXME rewrite so that it accepts only property_writer or splits and termination_annotations
        if property_writer is None and (
            splits is None or termination_annotations is None
        ):
            raise ValueError(
                "property_writer is not set, splits and termination_annotations must be set."
            )

        self.array = ts_array
        if property_writer is not None:
            _bbox_df, _splits, _termination_annotations, _attrs = property_writer.read()
        elif bboxes_df is None:
            _bbox_df = to_bbox_df(ts_array)
        else:
            _attrs = {}
        _bbox_df = bboxes_df if bboxes_df is not None else _bbox_df
        self._bboxes_dict = _bbox_df_to_dict(_bbox_df)
        self._safe_label = max(self._bboxes_dict.keys()) + 1

        self.splits = splits if splits is not None else _splits
        self.termination_annotations = (
            termination_annotations
            if termination_annotations is not None
            else _termination_annotations
        )
        self.property_writer = property_writer
        self.attrs = attrs if attrs is not None else _attrs

    def is_valid(self):
        """Check the consistency of the current TrackArray data.

        This function verifies that the bounding box DataFrame generated directly from the
        array matches the one constructed from the internal bounding box dictionary,
        and checks that each group's index is sorted.

        Returns
        -------
        bool
            True if the bounding box data is consistent and sorted, otherwise False.
        """
        _bboxes_df1 = to_bbox_df(self.array)
        _bboxes_df2 = _bbox_dict_to_df(self._bboxes_dict)
        is_all_sorted = all(
            [_df.index.is_monotonic_increasing for _df in self._bboxes_dict.values()]
        )
        return (
            _bboxes_df1.sort_values(["frame", "label"])
            .set_index(["frame", "label"])
            .equals(
                _bboxes_df2.sort_values(["frame", "label"]).set_index(
                    ["frame", "label"]
                )
            )
            & is_all_sorted
        )

    def write_properties(self):
        """
        Write the current properties using the associated property_writer.

        Calls the write method of the property_writer with the current bounding box DataFrame,
        splits, termination annotations, and additional attributes.

        Raises
        ------
        ValueError
            If property_writer is not set.
        """
        if self.property_writer is not None:
            self.property_writer.write(
                _bbox_dict_to_df(self._bboxes_dict),
                self.splits,
                self.termination_annotations,
                self.attrs,
            )
        else:
            raise ValueError("property_writer is not set, cannot write properties.")

    def _update_safe_label(self, new_label):
        """Update the safe label counter to ensure a unique track ID.

        Parameters
        ----------
        new_label : int
            The new label to consider when updating the safe label counter.
        """
        self._safe_label = max(self._safe_label, new_label + 1)

    def _get_track_bboxes(self, trackid: int):
        """Retrieve the bounding box DataFrame for a given track.

        Parameters
        ----------
        trackid : int
            The track identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame containing bounding box information for the specified track.
        """
        return self._bboxes_dict.get(trackid, pd.DataFrame())

    def _get_safe_track_id(self):
        """Get a safe (unused) track ID.

        Returns
        -------
        int
            A track ID that is safe to use for new tracks.
        """
        return self._safe_label

    def _get_bboxes(self, frames: Sequence[int], trackid: int):
        """Retrieve bounding boxes for specified frames and track ID.

        Parameters
        ----------
        frames : Sequence[int]
            A sequence of frame indices.
        trackid : int
            The track identifier.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'min_y', 'min_x', 'max_y', and 'max_x' for the specified frames.
        """
        rows = self._bboxes_dict[trackid].loc[frames]
        return rows[["min_y", "min_x", "max_y", "max_x"]]

    def __update_trackids_in_bboxes(self, frames, old_trackid, new_trackid):
        """Internal helper to update track IDs in the bounding box dictionary.

        Moves rows corresponding to the specified frames from old_trackid to new_trackid
        and removes the entry for old_trackid if it becomes empty.

        Parameters
        ----------
        frames : Iterable
            Frames whose bounding boxes should be updated.
        old_trackid : int
            The original track ID.
        new_trackid : int
            The new track ID to assign.
        """
        previous_rows = self._bboxes_dict[old_trackid].loc[frames]
        self._bboxes_dict[new_trackid] = pd.concat(
            [self._get_track_bboxes(new_trackid), previous_rows]
        ).sort_index()
        self._bboxes_dict[old_trackid].drop(index=frames, inplace=True)
        if self._bboxes_dict[old_trackid].empty:
            self._bboxes_dict.pop(old_trackid)

    def _update_trackids(
        self,
        frames: Sequence[int],
        trackid: int,
        new_trackid: int,
        txn: ts.Transaction,
    ):
        """Update track IDs in the array and bounding boxes.

        For the specified frames, updates the underlying tensorstore array by replacing
        the old track ID with the new track ID and updates the internal bounding box
        dictionary.

        Parameters
        ----------
        frames : Sequence[int]
            A sequence of frame indices where the update should occur.
        trackid : int
            The current track ID.
        new_trackid : int
            The new track ID to replace with.
        txn : ts.Transaction
            The transaction object to use for the update.

        Raises
        ------
        ValueError
            If the new track ID already exists in the bounding boxes for the specified frames.
        """
        if not set(frames).isdisjoint(self._get_track_bboxes(new_trackid).index):
            raise ValueError(
                f"new_trackid {new_trackid} already exists in the bboxes at frame {frames}"
            )

        array_txn = self.array.with_transaction(txn)
        rows = self._get_bboxes(frames, trackid)
        min_ys, min_xs, max_ys, max_xs = rows[
            ["min_y", "min_x", "max_y", "max_x"]
        ].values.T
        for frame, min_y, min_x, max_y, max_x in zip(
            frames, min_ys, min_xs, max_ys, max_xs
        ):
            subarr = array_txn[frame, min_y:max_y, min_x:max_x]
            ind = np.array(subarr) == trackid
            subarr[ts.d[:].translate_to[0]][ind] = new_trackid
            # Replace the trackid with the new_trackid

        self.__update_trackids_in_bboxes(frames, trackid, new_trackid)
        self._update_safe_label(new_trackid)

    def _cleanup_track_as_daughter(self, trackid: int):
        """Clean up splits for a track when it is treated as a daughter track.

        Removes the trackid from parent's daughter list in splits and then cleans up
        any single-daughter splits.

        Parameters
        ----------
        trackid : int
            The track identifier to clean up.
        """
        _splits = self.splits.copy()
        for parent, daughters in _splits.items():
            if int(trackid) in daughters:
                self.splits[int(parent)] = [
                    int(daughter) for daughter in daughters if daughter != trackid
                ]
        self.cleanup_single_daughter_splits()

    def _cleanup_track_as_parent(self, trackid: int):
        """Clean up data when a track is treated as a parent track.

        Removes the track from termination annotations and splits.

        Parameters
        ----------
        trackid : int
            The track identifier to clean up.
        """
        self.termination_annotations.pop(trackid, None)
        self.splits.pop(trackid, None)

    def delete_mask(
        self,
        frame: int,
        trackid: int,
        txn: ts.Transaction,
        cleanup: bool = True,
    ):
        """Delete the mask for a given track at a specified frame.

        This function removes the track's mask from the tensorstore array by setting
        the relevant pixels to 0, updates the bounding box dictionary, and, if required,
        performs cleanup of splits and termination annotations.

        Parameters
        ----------
        frame : int
            The frame index from which to delete the mask.
        trackid : int
            The track identifier whose mask is to be deleted.
        txn : ts.Transaction
            The transaction object for modifying the tensorstore array.
        cleanup : bool, optional
            Whether to perform cleanup if the track becomes empty (default is True).
        """
        row = self._get_bboxes([frame], trackid).iloc[0]
        min_y, min_x, max_y, max_x = row[["min_y", "min_x", "max_y", "max_x"]]
        array_txn = self.array.with_transaction(txn)
        subarr = array_txn[frame, min_y:max_y, min_x:max_x]
        ind = np.array(subarr) == trackid
        subarr[ts.d[:].translate_to[0]][ind] = 0
        self._bboxes_dict[trackid].drop(index=(frame), inplace=True)
        if (
            cleanup and self._get_track_bboxes(trackid).empty
        ):  # if the track becomes empty
            self._cleanup_track_as_daughter(trackid)
            self._cleanup_track_as_parent(trackid)

    def add_mask(
        self,
        frame: int,
        trackid: int,
        mask_origin: Sequence[int],
        mask,
        txn: ts.Transaction,
    ):
        """Add a mask for a given track at a specified frame.

        Inserts the mask into the tensorstore array and updates
        the internal bounding box dictionary accordingly.
        Also adjusts bounding boxes for any possibly updated labels.

        Parameters
        ----------
        frame : int
            The frame index where the mask is added.
        trackid : int
            The track identifier to update.
        mask_origin : Sequence[int]
            The top-left coordinate (origin) at which the mask should be applied.
        mask : array-like
            A boolean mask indicating the region to be updated.
        txn : ts.Transaction
            The transaction object for the tensorstore array.
        """
        assert mask.shape[0] + mask_origin[0] <= self.array.shape[1]
        assert mask.shape[1] + mask_origin[1] <= self.array.shape[2]
        assert mask.dtype == bool

        previous_frames = self._get_track_bboxes(trackid).index

        array_txn = self.array.with_transaction(txn)
        inds = np.where(mask)
        mask_min_y, mask_min_x = np.min(inds, axis=1)
        mask_max_y, mask_max_x = np.max(inds, axis=1)
        y_window = (mask_origin[0] + mask_min_y, mask_origin[0] + mask_max_y + 1)
        x_window = (mask_origin[1] + mask_min_x, mask_origin[1] + mask_max_x + 1)
        mask2 = mask[mask_min_y : mask_max_y + 1, mask_min_x : mask_max_x + 1]
        possibly_updated_labels = np.unique(
            array_txn[frame, y_window[0] : y_window[1], x_window[0] : x_window[1]][
                ts.d[:].translate_to[0]
            ][mask2]
        )
        possibly_updated_labels = set(possibly_updated_labels) - {0, trackid}
        array_txn[frame, y_window[0] : y_window[1], x_window[0] : x_window[1]][
            ts.d[:].translate_to[0]
        ][mask2] = trackid

        # Add entry to the bboxes_df
        self._bboxes_dict[trackid] = pd.concat(
            [
                self._bboxes_dict.get(trackid, pd.DataFrame()),
                pd.DataFrame(
                    {
                        "min_y": y_window[0],
                        "min_x": x_window[0],
                        "max_y": y_window[1],
                        "max_x": x_window[1],
                    },
                    index=pd.Index([frame], name="frame"),
                ),
            ]
        ).sort_index()

        # Update the bboxes_df for the possibly updated labels by overlapping with the new mask
        for updated_label in possibly_updated_labels:
            min_y, min_x, max_y, max_x = self._get_bboxes([frame], updated_label).iloc[
                0
            ][["min_y", "min_x", "max_y", "max_x"]]
            sublabel = array_txn[frame, min_y:max_y, min_x:max_x]
            ind = np.nonzero(np.array(sublabel) == updated_label)
            if np.any(ind):
                self._bboxes_dict[updated_label].loc[frame, "min_y"] = min_y + np.min(
                    ind[0]
                )
                self._bboxes_dict[updated_label].loc[frame, "min_x"] = min_x + np.min(
                    ind[1]
                )
                self._bboxes_dict[updated_label].loc[frame, "max_y"] = (
                    min_y + np.max(ind[0]) + 1
                )
                self._bboxes_dict[updated_label].loc[frame, "max_x"] = (
                    min_x + np.max(ind[1]) + 1
                )
            else:
                self._bboxes_dict[updated_label].drop(index=frame, inplace=True)
            if self._get_track_bboxes(updated_label).empty:
                self._cleanup_track_as_parent(updated_label)
                self._cleanup_track_as_daughter(updated_label)
        # Update splits and termination_annotations
        # invalidate splits and termination_annotations if
        # the frame is later than the last frame of the original track
        if len(previous_frames) > 0:
            min_frame = previous_frames.values[0]
            max_frame = previous_frames.values[-1]

            if frame > max_frame:
                self._cleanup_track_as_parent(trackid)
            # invalidate splits if the frame is earlier
            # than the first frame of the original track
            if frame < min_frame:
                self._cleanup_track_as_daughter(trackid)
        self._update_safe_label(trackid)

    def update_mask(
        self,
        frame: int,
        trackid: int,
        new_mask_origin: Sequence[int],
        new_mask,
        txn: ts.Transaction,
    ):
        """Update the mask for a given track at a specified frame.

        This function first deletes the existing mask and then adds the new mask.

        Parameters
        ----------
        frame : int
            The frame index to update.
        trackid : int
            The track identifier.
        new_mask_origin : Sequence[int]
            The origin where the new mask should be applied.
        new_mask : array-like
            The new boolean mask.
        txn : ts.Transaction
            The transaction object for the tensorstore update.
        """
        self.delete_mask(frame, trackid, txn, cleanup=False)
        self.add_mask(frame, trackid, new_mask_origin, new_mask, txn)

    def terminate_track(
        self, frame: int, trackid: int, annotation: str, txn: ts.Transaction
    ):
        """Terminate a track from a specified frame onward.

        Deletes the mask for every frame after the specified frame
        and sets the termination annotation.
        Also removes any splits associated with the track.

        Parameters
        ----------
        frame : int
            The frame index from which to terminate the track.
        trackid : int
            The track identifier.
        annotation : str
            A string annotation describing the termination.
        txn : ts.Transaction
            The transaction object for the tensorstore update.
        """
        bboxes_df = self._get_track_bboxes(trackid)
        bboxes_df = bboxes_df[bboxes_df.index > frame]
        for frame in bboxes_df.index:
            self.delete_mask(frame, trackid, txn)
        self.termination_annotations[trackid] = annotation
        self.splits.pop(int(trackid), None)

    def break_track(
        self,
        new_start_frame: int,
        trackid: int,
        change_after: bool,
        txn: ts.Transaction,
        new_trackid: Optional[int] = None,
    ):
        """Break a track into separate segments at a given start frame.

        Depending on the flag 'change_after', the function either updates the track for frames
        after the break or before the break. It also handles splits and termination annotations.

        Parameters
        ----------
        new_start_frame : int
            The frame index at which the track should be split.
        trackid : int
            The original track identifier.
        change_after : bool
            If True, modify the track for frames after new_start_frame.
            Otherwise, modify the track for frames before new_start_frame.
        txn : ts.Transaction
            The transaction object for the tensorstore update.
        new_trackid : Optional[int], optional
            A new track ID to assign; if None, a safe track ID will be generated.

        Returns
        -------
        int
            The new track ID assigned.

        Raises
        ------
        ValueError
            If the new track ID already exists in the bounding boxes.
        """
        if new_trackid is None:
            new_trackid = self._get_safe_track_id()
        bboxes_df = self._get_track_bboxes(trackid)
        if change_after:
            change_bboxes_df = bboxes_df.loc[new_start_frame:]
        else:
            change_bboxes_df = bboxes_df.loc[: new_start_frame - 1]

        if not set(change_bboxes_df.index).isdisjoint(
            self._get_track_bboxes(new_trackid).index
        ):
            raise ValueError("new_trackid already exists in the bboxes_df")

        frame_min = bboxes_df.index.values[0]
        frame_max = bboxes_df.index.values[-1]
        # Add the "break point" to the splits
        if frame_min == new_start_frame:
            # Delete the splits for which this track is a daughter
            self._cleanup_track_as_daughter(trackid)
        if frame_max + 1 == new_start_frame:
            # Delete the splits for which this track is a parent
            self._cleanup_track_as_parent(trackid)

        self._update_trackids(change_bboxes_df.index, trackid, new_trackid, txn)

        if change_after:
            # Update splits
            if trackid in self.splits:
                if new_trackid in self.splits:
                    raise ValueError("new_trackid already exists in splits")
                daughters = self.splits.pop(int(trackid))
                self.splits[int(new_trackid)] = daughters
            # Update termination_annotations
            if trackid in self.termination_annotations:
                self.termination_annotations[int(new_trackid)] = (
                    self.termination_annotations.pop(int(trackid))
                )
        else:
            # Update splits
            _splits = self.splits.copy()
            for parent, daughters in _splits.items():
                if trackid in daughters:
                    daughters.remove(int(trackid))
                    daughters.append(int(new_trackid))
                    self.splits[int(parent)] = daughters

        return new_trackid

    def add_split(
        self,
        daughter_start_frame: int,
        parent_trackid,
        daughter_trackids,
        txn: ts.Transaction,
    ):
        """Add a split to the tracks, breaking the parent track
        and updating daughters.

        Breaks the parent track at the specified daughter_start_frame
        and updates the splits for the parent and daughter tracks accordingly.

        Parameters
        ----------
        daughter_start_frame : int
            The frame index at which the split should occur.
        parent_trackid : int
            The track identifier for the parent track.
        daughter_trackids : list
            A list of track identifiers for daughter tracks.
        txn : ts.Transaction
            The transaction object for the tensorstore update.
        """
        new_trackid = self.break_track(
            daughter_start_frame, parent_trackid, change_after=True, txn=txn
        )
        daughter_trackids = [
            int(i) if i != parent_trackid else new_trackid for i in daughter_trackids
        ]
        for daughter_trackid in daughter_trackids:
            self.break_track(
                daughter_start_frame, daughter_trackid, change_after=False, txn=txn
            )
        self.splits[int(parent_trackid)] = daughter_trackids
        self.cleanup_single_daughter_splits()

    def cleanup_single_daughter_splits(self):
        """Clean up splits where a parent track has only a single daughter.

        In such cases, the split is merged back by updating track IDs
        and removing the split entry.
        """
        _splits = self.splits.copy()
        for parent, daughters in _splits.items():
            if len(daughters) == 1:
                daughter = int(daughters[0])
                track_df = self._get_track_bboxes(daughter)
                self._update_trackids(track_df.index, daughter, parent, None)
                self.splits.pop(int(parent))
