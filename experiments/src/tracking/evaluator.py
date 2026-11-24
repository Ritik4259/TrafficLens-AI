"""
Evaluation engine for multi-object tracking against UA-DETRAC ground truth.

Integrates TrackEval (official HOTA, DetA, AssA) and motmetrics (CLEAR MOTA,
IDF1, IDSW, MT, ML, Frag), respecting UA-DETRAC ignore region semantics.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# NumPy 2.x compatibility patch for TrackEval
np.float = float
np.int = int

PROJECT_ROOT = Path(__file__).resolve().parents[3]
trackeval_path = PROJECT_ROOT / "experiments" / "third_party" / "TrackEval"
if str(trackeval_path) not in sys.path:
    sys.path.insert(0, str(trackeval_path))

from trackeval.metrics import HOTA
import motmetrics as mm

from experiments.src.dataset.parser import BBox, SequenceAnnotation
from .config import TrackingConfig, TrackingEvaluationConfig
from .tracker import Track, VehicleTracker


class TrackingEvaluator:
    """Evaluates multi-object tracking performance using standardized HOTA and MOTA metrics."""

    def __init__(self, config: Optional[TrackingEvaluationConfig] = None):
        self.config = config or TrackingEvaluationConfig()
        self.hota_evaluator = HOTA()

    def evaluate_sequence(
        self,
        tracks_by_frame: Dict[int, List[Track]],
        seq_ann: SequenceAnnotation,
    ) -> Dict[str, Any]:
        """
        Evaluates tracking results for a single sequence:
        - Excludes unmatched tracks within ignore regions to avoid unfair FP penalties.
        - Computes HOTA, DetA, AssA, LocA using official TrackEval.
        - Computes MOTA, IDF1, IDSW, MT, ML, Frag using motmetrics.
        """
        acc = mm.MOTAccumulator(auto_id=True)
        ignored_region = seq_ann.ignored_region

        all_frame_nums = sorted(set(tracks_by_frame.keys()) | set(seq_ann.frames.keys()))
        num_timesteps = len(all_frame_nums)

        # Preprocessing structures for TrackEval HOTA
        seq_gt_ids: List[np.ndarray] = []
        seq_trk_ids: List[np.ndarray] = []
        seq_gt_dets: List[np.ndarray] = []
        seq_trk_dets: List[np.ndarray] = []
        seq_similarities: List[np.ndarray] = []

        total_ignored_track_dets = 0
        raw_tracker_dets_count = 0
        kept_tracker_dets_count = 0
        total_gt_dets_count = 0

        unique_gt_id_set = set()
        unique_trk_id_set = set()

        for f_idx in all_frame_nums:
            f_ann = seq_ann.frames.get(f_idx, None)
            gt_targets = f_ann.targets if f_ann else []
            gt_ids = [t.target_id for t in gt_targets]
            gt_boxes = [t.bbox for t in gt_targets]
            total_gt_dets_count += len(gt_boxes)
            unique_gt_id_set.update(gt_ids)

            raw_tracks = tracks_by_frame.get(f_idx, [])
            raw_tracker_dets_count += len(raw_tracks)

            # Filter tracks using ignore-region semantics:
            # 1. Matches against valid GT boxes are always kept.
            # 2. Unmatched tracks overlapping ignore regions >= threshold are excluded.
            kept_tracks: List[Track] = []
            for trk in raw_tracks:
                # Check match against any GT
                matched_gt = False
                for gb in gt_boxes:
                    if trk.bbox.iou(gb) >= self.config.iou_threshold:
                        matched_gt = True
                        break

                if matched_gt:
                    kept_tracks.append(trk)
                else:
                    # Check if unmatched track falls in ignore region
                    if ignored_region and ignored_region.is_box_ignored(
                        trk.bbox, threshold=self.config.ignore_overlap_threshold
                    ):
                        total_ignored_track_dets += 1
                    else:
                        kept_tracks.append(trk)

            kept_tracker_dets_count += len(kept_tracks)
            trk_ids = [t.track_id for t in kept_tracks]
            trk_boxes = [t.bbox for t in kept_tracks]
            unique_trk_id_set.update(trk_ids)

            # Compute pairwise distance matrix (1 - IoU) for motmetrics
            dist_matrix = np.full((len(gt_boxes), len(trk_boxes)), np.nan)
            sim_matrix = np.zeros((len(gt_boxes), len(trk_boxes)), dtype=np.float32)

            for g_i, gb in enumerate(gt_boxes):
                for t_i, tb in enumerate(trk_boxes):
                    iou = gb.iou(tb)
                    sim_matrix[g_i, t_i] = iou
                    if iou >= self.config.iou_threshold:
                        dist_matrix[g_i, t_i] = 1.0 - iou

            acc.update(gt_ids, trk_ids, dist_matrix)

            # Record per-frame arrays for TrackEval
            seq_gt_ids.append(np.array(gt_ids, dtype=np.int32))
            seq_trk_ids.append(np.array(trk_ids, dtype=np.int32))
            seq_gt_dets.append(
                np.array([b.to_xyxy() for b in gt_boxes], dtype=np.float32)
                if gt_boxes else np.empty((0, 4), dtype=np.float32)
            )
            seq_trk_dets.append(
                np.array([b.to_xyxy() for b in trk_boxes], dtype=np.float32)
                if trk_boxes else np.empty((0, 4), dtype=np.float32)
            )
            seq_similarities.append(sim_matrix)

        # 1. Compute motmetrics summary
        mh = mm.metrics.create()
        mot_summary = mh.compute(
            acc,
            metrics=[
                "mota", "motp", "idf1", "idp", "idr", "recall", "precision",
                "num_unique_objects", "mostly_tracked", "partially_tracked",
                "mostly_lost", "num_false_positives", "num_misses",
                "num_switches", "num_fragmentations",
            ],
            name=seq_ann.name,
        )

        mota = float(mot_summary["mota"].iloc[0])
        motp = float(mot_summary["motp"].iloc[0])
        idf1 = float(mot_summary["idf1"].iloc[0])
        idp = float(mot_summary["idp"].iloc[0])
        idr = float(mot_summary["idr"].iloc[0])
        recall = float(mot_summary["recall"].iloc[0])
        precision = float(mot_summary["precision"].iloc[0])
        idsw = int(mot_summary["num_switches"].iloc[0])
        frag = int(mot_summary["num_fragmentations"].iloc[0])
        mt = int(mot_summary["mostly_tracked"].iloc[0])
        pt = int(mot_summary["partially_tracked"].iloc[0])
        ml = int(mot_summary["mostly_lost"].iloc[0])
        fp = int(mot_summary["num_false_positives"].iloc[0])
        fn = int(mot_summary["num_misses"].iloc[0])

        # 2. Compute TrackEval HOTA
        # Map IDs to contiguous 0-indexed integers
        gt_id_map = {orig_id: idx for idx, orig_id in enumerate(sorted(unique_gt_id_set))}
        trk_id_map = {orig_id: idx for idx, orig_id in enumerate(sorted(unique_trk_id_set))}

        contiguous_gt_ids = [
            np.array([gt_id_map[i] for i in frame_ids], dtype=np.int32) if len(frame_ids) > 0 else np.empty(0, dtype=np.int32)
            for frame_ids in seq_gt_ids
        ]
        contiguous_trk_ids = [
            np.array([trk_id_map[i] for i in frame_ids], dtype=np.int32) if len(frame_ids) > 0 else np.empty(0, dtype=np.int32)
            for frame_ids in seq_trk_ids
        ]

        hota_data = {
            "num_timesteps": num_timesteps,
            "num_gt_ids": len(unique_gt_id_set),
            "num_tracker_ids": len(unique_trk_id_set),
            "num_gt_dets": total_gt_dets_count,
            "num_tracker_dets": kept_tracker_dets_count,
            "gt_ids": contiguous_gt_ids,
            "tracker_ids": contiguous_trk_ids,
            "gt_dets": seq_gt_dets,
            "tracker_dets": seq_trk_dets,
            "similarity_scores": seq_similarities,
        }

        hota_res = self.hota_evaluator.eval_sequence(hota_data)

        hota_score = float(np.mean(hota_res["HOTA"]))
        deta_score = float(np.mean(hota_res["DetA"]))
        assa_score = float(np.mean(hota_res["AssA"]))
        loca_score = float(np.mean(hota_res["LocA"]))

        return {
            "sequence_name": seq_ann.name,
            "condition": seq_ann.scene_condition,
            "camera_state": seq_ann.camera_state,
            "total_frames": num_timesteps,
            "hota": hota_score,
            "deta": deta_score,
            "assa": assa_score,
            "loca": loca_score,
            "mota": mota,
            "motp": motp,
            "idf1": idf1,
            "idp": idp,
            "idr": idr,
            "precision": precision,
            "recall": recall,
            "idsw": idsw,
            "frag": frag,
            "mostly_tracked": mt,
            "partially_tracked": pt,
            "mostly_lost": ml,
            "fp": fp,
            "fn": fn,
            "total_gt_vehicles": len(unique_gt_id_set),
            "total_gt_detections": total_gt_dets_count,
            "total_tracker_ids": len(unique_trk_id_set),
            "total_tracker_detections": kept_tracker_dets_count,
            "ignored_track_detections": total_ignored_track_dets,
        }

    def evaluate_dataset(
        self,
        tracks_by_seq: Dict[str, Dict[int, List[Track]]],
        sequences: Dict[str, SequenceAnnotation],
    ) -> Dict[str, Any]:
        """Evaluates multiple sequences and aggregates tracking metrics."""
        per_seq_results = []
        all_hota = []
        all_deta = []
        all_assa = []
        all_mota = []
        all_idf1 = []
        all_idsw = 0
        all_frag = 0
        all_mt = 0
        all_ml = 0
        all_fp = 0
        all_fn = 0
        all_gt_boxes = 0
        all_ignored_dets = 0

        for seq_name, tracks_by_frame in tracks_by_seq.items():
            seq_ann = sequences[seq_name]
            res = self.evaluate_sequence(tracks_by_frame, seq_ann)
            per_seq_results.append(res)

            all_hota.append(res["hota"])
            all_deta.append(res["deta"])
            all_assa.append(res["assa"])
            all_mota.append(res["mota"])
            all_idf1.append(res["idf1"])
            all_idsw += res["idsw"]
            all_frag += res["frag"]
            all_mt += res["mostly_tracked"]
            all_ml += res["mostly_lost"]
            all_fp += res["fp"]
            all_fn += res["fn"]
            all_gt_boxes += res["total_gt_detections"]
            all_ignored_dets += res["ignored_track_detections"]

        return {
            "mean_hota": float(np.mean(all_hota)) if all_hota else 0.0,
            "mean_deta": float(np.mean(all_deta)) if all_deta else 0.0,
            "mean_assa": float(np.mean(all_assa)) if all_assa else 0.0,
            "mean_mota": float(np.mean(all_mota)) if all_mota else 0.0,
            "mean_idf1": float(np.mean(all_idf1)) if all_idf1 else 0.0,
            "total_idsw": all_idsw,
            "total_frag": all_frag,
            "total_mostly_tracked": all_mt,
            "total_mostly_lost": all_ml,
            "total_fp": all_fp,
            "total_fn": all_fn,
            "total_gt_detections": all_gt_boxes,
            "total_ignored_detections": all_ignored_dets,
            "per_sequence": per_seq_results,
        }

    def sweep_parameters(
        self,
        predictions_by_seq: Dict[str, Dict[int, List[Prediction]]],
        sequences: Dict[str, SequenceAnnotation],
        high_thresholds: Optional[List[float]] = None,
        match_thresholds: Optional[List[float]] = None,
        track_buffers: Optional[List[int]] = None,
    ) -> Tuple[TrackingConfig, List[Dict[str, Any]]]:
        """
        Sweeps ByteTrack parameters across validation sequences to identify
        the configuration that maximizes HOTA and MOTA.
        """
        if high_thresholds is None:
            high_thresholds = [0.40, 0.50, 0.60]
        if match_thresholds is None:
            match_thresholds = [0.70, 0.80]
        if track_buffers is None:
            track_buffers = [20, 30]

        sweep_records = []
        best_hota = -1.0
        best_config = TrackingConfig()

        for high_th in high_thresholds:
            for match_th in match_thresholds:
                for buf in track_buffers:
                    cfg = TrackingConfig(
                        track_high_thresh=high_th,
                        track_low_thresh=0.10,
                        new_track_thresh=high_th + 0.10,
                        track_buffer=buf,
                        match_thresh=match_th,
                    )
                    tracker = VehicleTracker(cfg)

                    val_tracks_by_seq = {}
                    for seq_name, preds_by_frame in predictions_by_seq.items():
                        tracks, _ = tracker.track_sequence(preds_by_frame)
                        val_tracks_by_seq[seq_name] = tracks

                    res = self.evaluate_dataset(val_tracks_by_seq, sequences)
                    record = {
                        "track_high_thresh": high_th,
                        "track_low_thresh": 0.10,
                        "match_thresh": match_th,
                        "track_buffer": buf,
                        "mean_hota": res["mean_hota"],
                        "mean_deta": res["mean_deta"],
                        "mean_assa": res["mean_assa"],
                        "mean_mota": res["mean_mota"],
                        "mean_idf1": res["mean_idf1"],
                        "total_idsw": res["total_idsw"],
                    }
                    sweep_records.append(record)

                    if res["mean_hota"] > best_hota:
                        best_hota = res["mean_hota"]
                        best_config = cfg

        return best_config, sweep_records
