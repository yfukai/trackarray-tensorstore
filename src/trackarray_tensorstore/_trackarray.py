from typing import Optional
from typing import Sequence
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorstore as ts
from numpy import typing as npt
from skimage.measure import regionprops_table


def to_bbox_df(label: npt.ArrayLike) -> pd.DataFrame:
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
    # Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
    if bbox_df.empty:
        return bbox_df
    bbox_df["min_y"] = bbox_df["bbox-0"]
    bbox_df["min_x"] = bbox_df["bbox-1"]
    bbox_df["max_y"] = bbox_df["bbox-2"]
    bbox_df["max_x"] = bbox_df["bbox-3"]
    del bbox_df["bbox-0"], bbox_df["bbox-1"], bbox_df["bbox-2"], bbox_df["bbox-3"]
    return bbox_df


def _bbox_df_to_dict(bboxes_df):
    return {
        label: grp.set_index("frame").sort_index()
        for label, grp in bboxes_df.groupby("label")
    }


def _bbox_dict_to_df(bboxes_dict):
    return pd.concat(
        [grp.assign(label=label).reset_index() for label, grp in bboxes_dict.items()]
    ).reset_index(drop=True)


class TrackArray:
    def __init__(
        self,
        ts_array: ts.TensorStore,
        splits: Optional[Dict[int, List[int]]] = None,
        termination_annotations: Optional[Dict[int, str]]=None,
        bboxes_df=None,*,
        property_writer=None,
        attrs = None
    ):
        # FIXME rewrite so that it accepts only property_writer or splits and termination_annotations
        if property_writer is None and (splits is None or termination_annotations is None):
            raise ValueError("property_writer is not set, splits and termination_annotations must be set.")
        
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
        self.termination_annotations = termination_annotations if termination_annotations is not None else _termination_annotations
        self.property_writer = property_writer
        self.attrs = attrs if attrs is not None else _attrs

    def is_valid(self):
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
        if self.property_writer is not None:
            self.property_writer.write(
                self._bboxes_dict, self.splits, self.termination_annotations, self.attrs
            )
        else:
            raise ValueError("property_writer is not set, cannot write properties.")

    def _update_safe_label(self, new_label):
        self._safe_label = max(self._safe_label, new_label + 1)

    def _get_track_bboxes(self, trackid: int):
        return self._bboxes_dict.get(trackid, pd.DataFrame())

    def _get_safe_track_id(self):
        return self._safe_label

    def _get_bboxes(self, frames: Sequence[int], trackid: int):
        rows = self._bboxes_dict[trackid].loc[frames]
        return rows[["min_y", "min_x", "max_y", "max_x"]]

    def __update_trackids_in_bboxes(self, frames, old_trackid, new_trackid):
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
        _splits = self.splits.copy()
        for parent, daughters in _splits.items():
            if int(trackid) in daughters:
                self.splits[int(parent)] = [
                    int(daughter) for daughter in daughters if daughter != trackid
                ]
        self.cleanup_single_daughter_splits()

    def _cleanup_track_as_parent(self, trackid: int):
        self.termination_annotations.pop(trackid, None)
        self.splits.pop(trackid, None)

    def delete_mask(
        self,
        frame: int,
        trackid: int,
        txn: ts.Transaction,
        cleanup: bool = True,
    ):
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
        # invalidate splits and termination_annotations if the frame is later than the last frame of the original track
        if len(previous_frames) > 0:
            min_frame = previous_frames.values[0]
            max_frame = previous_frames.values[-1]

            if frame > max_frame:
                self._cleanup_track_as_parent(trackid)
            # invalidate splits if the frame is earlier than the first frame of the original track
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
        self.delete_mask(frame, trackid, txn, cleanup=False)
        self.add_mask(frame, trackid, new_mask_origin, new_mask, txn)

    def terminate_track(
        self, frame: int, trackid: int, annotation: str, txn: ts.Transaction
    ):
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
                self.termination_annotations[
                    int(new_trackid)
                ] = self.termination_annotations.pop(int(trackid))
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
        _splits = self.splits.copy()
        for parent, daughters in _splits.items():
            if len(daughters) == 1:
                daughter = int(daughters[0])
                track_df = self._get_track_bboxes(daughter)
                self._update_trackids(track_df.index, daughter, parent, None)
                self.splits.pop(int(parent))
