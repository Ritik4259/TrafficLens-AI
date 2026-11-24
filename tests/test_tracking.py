"""
Unit tests for ByteTrack vehicle tracking pipeline, lifecycle management, and evaluation.
"""

import pytest
import numpy as np
from pathlib import Path

from experiments.src.dataset.parser import BBox, IgnoredRegion, SequenceAnnotation, FrameAnnotation, Target, VehicleAttribute
from experiments.src.detection.metrics import Prediction
from experiments.src.tracking.config import TrackingConfig, TrackingEvaluationConfig
from experiments.src.tracking.tracker import Track, VehicleTracker
from experiments.src.tracking.exporter import MOTExporter
from experiments.src.tracking.evaluator import TrackingEvaluator


class TestTrackLifecycleAndPersistence:
    def test_track_id_persistence(self):
        """Vehicle moves continuously across consecutive frames; track ID must persist."""
        tracker = VehicleTracker(TrackingConfig(track_high_thresh=0.5, new_track_thresh=0.6))

        # Frame 1: Vehicle at (100, 100)
        p1 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.85)]
        t1 = tracker.update(p1)
        assert len(t1) == 1
        initial_id = t1[0].track_id

        # Frame 2: Vehicle moves slightly to (104, 102)
        p2 = [Prediction(bbox=BBox(104.0, 102.0, 50.0, 50.0), confidence=0.80)]
        t2 = tracker.update(p2)
        assert len(t2) == 1
        assert t2[0].track_id == initial_id

        # Frame 3: Vehicle moves to (108, 104)
        p3 = [Prediction(bbox=BBox(108.0, 104.0, 50.0, 50.0), confidence=0.82)]
        t3 = tracker.update(p3)
        assert len(t3) == 1
        assert t3[0].track_id == initial_id
        assert len(t3[0].history) == 3

    def test_new_track_creation_and_confirmation(self):
        """
        Second vehicle entering the scene is confirmed across consecutive frames
        and receives a distinct persistent track ID.
        """
        tracker = VehicleTracker(TrackingConfig(track_high_thresh=0.5, new_track_thresh=0.6))

        # Frame 1: Vehicle A
        p1 = [Prediction(bbox=BBox(50.0, 50.0, 40.0, 40.0), confidence=0.90)]
        t1 = tracker.update(p1)
        assert len(t1) == 1
        id_a = t1[0].track_id

        # Frame 2: Vehicle A and new Vehicle B (Vehicle B is unconfirmed in ByteTrack on frame 2)
        p2 = [
            Prediction(bbox=BBox(52.0, 52.0, 40.0, 40.0), confidence=0.90),
            Prediction(bbox=BBox(400.0, 300.0, 60.0, 60.0), confidence=0.88),
        ]
        t2 = tracker.update(p2)
        assert len(t2) == 1  # Only confirmed vehicle A emitted

        # Frame 3: Vehicle A and Vehicle B detected again -> Vehicle B confirms and activates
        p3 = [
            Prediction(bbox=BBox(54.0, 54.0, 40.0, 40.0), confidence=0.90),
            Prediction(bbox=BBox(402.0, 302.0, 60.0, 60.0), confidence=0.88),
        ]
        t3 = tracker.update(p3)
        assert len(t3) == 2
        ids = {t.track_id for t in t3}
        assert id_a in ids
        id_b = (ids - {id_a}).pop()
        assert id_b != id_a

    def test_temporary_occlusion_recovery(self):
        """Vehicle lost for 2 frames recovers original track ID upon reappearance."""
        tracker = VehicleTracker(TrackingConfig(track_buffer=10, match_thresh=0.8))

        # Frame 1: Vehicle visible
        p1 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)]
        t1 = tracker.update(p1)
        orig_id = t1[0].track_id

        # Frames 2 & 3: Occluded (no detections)
        t2 = tracker.update([])
        assert len(t2) == 0
        t3 = tracker.update([])
        assert len(t3) == 0

        # Frame 4: Reappears near predicted motion
        p4 = [Prediction(bbox=BBox(106.0, 106.0, 50.0, 50.0), confidence=0.85)]
        t4 = tracker.update(p4)
        assert len(t4) == 1
        assert t4[0].track_id == orig_id

    def test_track_termination_after_buffer(self):
        """Track is removed after missing for more than track_buffer frames."""
        tracker = VehicleTracker(TrackingConfig(track_buffer=3))

        # Frame 1: Active vehicle
        p1 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)]
        t1 = tracker.update(p1)
        orig_id = t1[0].track_id

        # Frames 2 to 7: Disappeared for 6 frames (> buffer 3)
        for _ in range(6):
            tracker.update([])

        # Frame 8 & 9: Detection reappears long after buffer expiry; must be confirmed as NEW track ID
        p8 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)]
        tracker.update(p8)
        p9 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)]
        t9 = tracker.update(p9)
        assert len(t9) == 1
        assert t9[0].track_id != orig_id


class TestEdgeCasesAndRobustness:
    def test_sequence_level_id_reset(self):
        """Calling tracker.reset() resets track IDs back to 1 for new sequences."""
        tracker = VehicleTracker()

        p1 = [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)]
        t1 = tracker.update(p1)
        assert t1[0].track_id == 1

        tracker.reset()
        p2 = [Prediction(bbox=BBox(200.0, 200.0, 50.0, 50.0), confidence=0.90)]
        t2 = tracker.update(p2)
        assert t2[0].track_id == 1

    def test_empty_detection_frames(self):
        """Empty detection frames do not throw errors."""
        tracker = VehicleTracker()
        tracks = tracker.update([])
        assert tracks == []

    def test_invalid_degenerate_bounding_boxes(self):
        """Boxes with zero or negative dimensions are filtered out safely."""
        tracker = VehicleTracker()
        invalid_preds = [
            Prediction(bbox=BBox(0.0, 0.0, 0.0, 0.0), confidence=0.90),
            Prediction(bbox=BBox(10.0, 10.0, -5.0, 20.0), confidence=0.85),
            Prediction(bbox=BBox(100.0, 100.0, 40.0, 40.0), confidence=0.80),
        ]
        tracks = tracker.update(invalid_preds)
        assert len(tracks) == 1
        assert tracks[0].bbox.width == 40.0

    def test_duplicate_detections_handling(self):
        """Two identical overlapping candidate detections do not produce duplicate active tracks."""
        tracker = VehicleTracker(TrackingConfig(nms_iou_threshold=0.50))
        duplicates = [
            Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90),
            Prediction(bbox=BBox(100.5, 100.5, 49.5, 49.5), confidence=0.85),
        ]
        tracks = tracker.update(duplicates)
        # Should initiate one track for this physical vehicle
        assert len(tracks) == 1

    def test_frame_ordering_preservation(self):
        """track_sequence processes frames in sorted chronological order."""
        tracker = VehicleTracker()
        preds_unordered = {
            3: [Prediction(bbox=BBox(106.0, 106.0, 50.0, 50.0), confidence=0.85)],
            1: [Prediction(bbox=BBox(100.0, 100.0, 50.0, 50.0), confidence=0.90)],
            2: [Prediction(bbox=BBox(103.0, 103.0, 50.0, 50.0), confidence=0.88)],
        }
        tracks_by_frame, stats = tracker.track_sequence(preds_unordered)
        assert sorted(tracks_by_frame.keys()) == [1, 2, 3]
        # Same ID across all 3 frames
        assert tracks_by_frame[1][0].track_id == tracks_by_frame[2][0].track_id == tracks_by_frame[3][0].track_id


class TestMOTExportAndEvaluation:
    def test_mot_exporter_roundtrip(self, tmp_path: Path):
        """Tracks exported to MOT TXT match the standard MOTChallenge schema."""
        tracks = {
            1: [Track(track_id=1, bbox=BBox(100.0, 150.0, 50.0, 60.0), confidence=0.88)],
            2: [Track(track_id=1, bbox=BBox(104.0, 152.0, 50.0, 60.0), confidence=0.89)],
        }
        mot_file = tmp_path / "tracks.txt"
        MOTExporter.export_tracks_to_mot_txt(tracks, mot_file)

        loaded_tracks = MOTExporter.load_mot_txt(mot_file)
        assert len(loaded_tracks[1]) == 1
        assert loaded_tracks[1][0].track_id == 1
        assert loaded_tracks[1][0].bbox.left == 100.0

    def test_evaluator_ignore_region_filtering(self):
        """Tracks falling into ignore regions without ground truth are not penalized as False Positives."""
        ign_box = BBox(left=0.0, top=0.0, width=200.0, height=200.0)
        ignored_region = IgnoredRegion(boxes=[ign_box])

        seq_ann = SequenceAnnotation(
            name="TEST_EVAL",
            camera_state="stable",
            scene_condition="daylight",
            fog_status="no_fog",
            ignored_region=ignored_region,
            frames={
                1: FrameAnnotation(
                    frame_num=1,
                    density=1,
                    targets=[Target(target_id=1, bbox=BBox(300.0, 300.0, 50.0, 50.0), attribute=VehicleAttribute(vehicle_type="car"))],
                )
            },
        )

        # Track 1 matches valid GT perfectly
        # Track 2 is an unmatched candidate inside the ignore region
        tracks_by_frame = {
            1: [
                Track(track_id=1, bbox=BBox(300.0, 300.0, 50.0, 50.0), confidence=0.90),
                Track(track_id=2, bbox=BBox(50.0, 50.0, 40.0, 40.0), confidence=0.75),
            ]
        }

        evaluator = TrackingEvaluator()
        res = evaluator.evaluate_sequence(tracks_by_frame, seq_ann)

        # Track 2 is excluded as ignored detection
        assert res["ignored_track_detections"] == 1
        assert res["fp"] == 0
        assert res["mota"] == 1.0
        assert res["hota"] == 1.0
