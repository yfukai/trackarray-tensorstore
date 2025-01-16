from typing import Optional
from typing import Sequence

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
    ).set_index(["frame", "label"])
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


class TrackArray:
    def __init__(self, ts_array, splits, termination_annotations, bboxes_df=None):
        self.array = ts_array
        if bboxes_df is None:
            bboxes_df = to_bbox_df(ts_array)
        self.bboxes_df = bboxes_df
        self.update_track_df()
        self.splits = splits
        self.termination_annotations = termination_annotations

    def update_track_df(self):
        self._track_df = (
            self.bboxes_df.reset_index().groupby("label")["frame"].agg(["min", "max"])
        )

    def is_valid(self):
        _bboxes_df = to_bbox_df(self.array)
        return _bboxes_df.sort_index().equals(self.bboxes_df.sort_index())

    def _get_track_bboxes(self, trackid: int):
        return self.bboxes_df[self.bboxes_df.index.get_level_values("label") == trackid]

    def _get_safe_track_id(self):
        return self.bboxes_df.index.get_level_values("label").max() + 1

    def __get_bbox(self, frame: int, trackid: int):
        row = self.bboxes_df.loc[(frame, trackid)]
        return row[["min_y", "min_x", "max_y", "max_x"]]

    def _update_trackid(
        self,
        frame: int,
        trackid: int,
        new_trackid: int,
        txn: ts.Transaction,
        skip_update=False,
    ):
        if (frame, new_trackid) in self.bboxes_df.index:
            raise ValueError("new_trackid already exists in the bboxes_df")

        array_txn = self.array.with_transaction(txn)
        min_y, min_x, max_y, max_x = self.__get_bbox(frame, trackid)
        subarr = array_txn[frame, min_y:max_y, min_x:max_x]
        ind = np.array(subarr) == trackid
        subarr[ts.d[:].translate_to[0]][
            ind
        ] = new_trackid  # Replace the trackid with the new_trackid
        self.bboxes_df.index = self.bboxes_df.index.map(
            lambda x: (frame, new_trackid) if x == (frame, trackid) else x
        )

        if not skip_update:
            self.update_track_df()

    def _cleanup_track(self, trackid: int):
        self.termination_annotations.pop(trackid, None)
        self.splits.pop(trackid, None)
        for parent, daughters in self.splits.copy().items():
            self.splits[int(parent)] = [
                int(daughter) for daughter in daughters if daughter != trackid
            ]
        self.cleanup_single_daughter_splits()

    def delete_mask(
        self,
        frame: int,
        trackid: int,
        txn: ts.Transaction,
        skip_update=False,
        cleanup=True,
    ):
        min_y, min_x, max_y, max_x = self.__get_bbox(frame, trackid)
        array_txn = self.array.with_transaction(txn)
        subarr = array_txn[frame, min_y:max_y, min_x:max_x]
        ind = np.array(subarr) == trackid
        subarr[ts.d[:].translate_to[0]][ind] = 0
        self.bboxes_df.drop(index=(frame, trackid), inplace=True)
        if not skip_update:
            self.update_track_df()
        if (
            cleanup and self._get_track_bboxes(trackid).empty
        ):  # if the track becomes empty
            self._cleanup_track(trackid)

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

        previous_frames = self._get_track_bboxes(trackid).reset_index().frame.copy()

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
        self.bboxes_df = pd.concat(
            [
                self.bboxes_df,
                pd.DataFrame(
                    {
                        "min_y": y_window[0],
                        "min_x": x_window[0],
                        "max_y": y_window[1],
                        "max_x": x_window[1],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [(frame, trackid)], names=["frame", "label"]
                    ),
                ),
            ]
        )

        # Update the bboxes_df for the possibly updated labels by overlapping with the new mask
        for updated_label in possibly_updated_labels:
            min_y, min_x, max_y, max_x = self.__get_bbox(frame, updated_label)
            sublabel = array_txn[frame, min_y:max_y, min_x:max_x]
            ind = np.nonzero(np.array(sublabel) == updated_label)
            if np.any(ind):
                self.bboxes_df.loc[(frame, updated_label), "min_y"] = min_y + np.min(
                    ind[0]
                )
                self.bboxes_df.loc[(frame, updated_label), "min_x"] = min_x + np.min(
                    ind[1]
                )
                self.bboxes_df.loc[(frame, updated_label), "max_y"] = (
                    min_y + np.max(ind[0]) + 1
                )
                self.bboxes_df.loc[(frame, updated_label), "max_x"] = (
                    min_x + np.max(ind[1]) + 1
                )
            else:
                self.bboxes_df.drop(index=(frame, updated_label), inplace=True)
            if self._get_track_bboxes(updated_label).empty:
                self._cleanup_track(updated_label)

        # Update splits and termination_annotations
        # invalidate splits and termination_annotations if the frame is later than the last frame of the original track
        if frame > previous_frames.max():
            self.termination_annotations.pop(trackid, None)
            self.splits.pop(trackid, None)
        # invalidate splits if the frame is earlier than the first frame of the original track
        if frame < previous_frames.min():
            _splits = self.splits.copy()
            for parent, daughters in _splits.items():
                self.splits[int(parent)] = [
                    int(daughter) for daughter in daughters if daughter != trackid
                ]
            self.cleanup_single_daughter_splits()

        self.update_track_df()

    def update_mask(
        self,
        frame: int,
        trackid: int,
        new_mask_origin: Sequence[int],
        new_mask,
        txn: ts.Transaction,
    ):
        self.delete_mask(frame, trackid, txn, cleanup=False, skip_update=True)
        self.add_mask(frame, trackid, new_mask_origin, new_mask, txn)

    def terminate_track(
        self, frame: int, trackid: int, annotation: str, txn: ts.Transaction
    ):
        bboxes_df = self._get_track_bboxes(trackid).reset_index()
        bboxes_df = bboxes_df[bboxes_df.frame > frame]
        for frame in bboxes_df.frame:
            self.delete_mask(frame, trackid, txn, skip_update=True)
        self.update_track_df()
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
        """"""
        if new_trackid is None:
            new_trackid = self._get_safe_track_id()
        bboxes_df = self._get_track_bboxes(trackid).reset_index()
        if change_after:
            change_bboxes_df = bboxes_df[bboxes_df.frame >= new_start_frame]
        else:
            change_bboxes_df = bboxes_df[bboxes_df.frame < new_start_frame]

        for frame in change_bboxes_df.frame:
            if (frame, new_trackid) in self.bboxes_df.index:
                raise ValueError("new_trackid already exists in the bboxes_df")

        # Add the "break point" to the splits
        if bboxes_df.frame.min() == new_start_frame:
            # Delete the splits for which this track is a daughter
            _splits = self.splits.copy()
            for parent, daughters in _splits.items():
                if trackid in daughters:
                    daughters.remove(int(trackid))
                    self.splits[int(parent)] = daughters
        if bboxes_df.frame.max() + 1 == new_start_frame:
            # Delete the splits for which this track is a parent
            self.splits.pop(int(trackid), None)

        for frame in change_bboxes_df.frame:
            self._update_trackid(frame, trackid, new_trackid, txn, skip_update=True)
        self.update_track_df()

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
        for parent, daughters in self.splits.copy().items():
            if len(daughters) == 1:
                daughter = int(daughters[0])
                track_df = self._get_track_bboxes(daughter).reset_index()
                for frame in track_df.frame:
                    self._update_trackid(
                        frame, daughter, parent, None, skip_update=True
                    )
                self.splits.pop(int(parent))
