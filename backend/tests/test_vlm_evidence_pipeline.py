"""
Unit and integration tests for the Multimodal Evidence Pipeline and Keyframe Extraction.

Validates:
- Deterministic extraction and tie-breaking
- Frame budget enforcement (K) and pruning
- Handling of fewer candidates than K
- Stopped-vehicle event prioritization over periodic anchors
- Temporal diversity and duplicate timestamp suppression
- Invalid, missing, and out-of-range timestamp robustness
- Missing FPS fallback handling
- Empty and corrupt video handling (VideoReadError)
- Failed frame decoding resilience
- Aspect-ratio preserving downscaling
- OpenCV resource cleanup
- EvidencePackager provenance and public schema path isolation
"""

import math
import tempfile
from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import pytest

from backend.app.schemas.vlm import (
    ExtractionConfigSchema,
    ExtractionDiagnosticsSchema,
    EvidencePackageSchema,
    EvidencePackagePublicSchema,
)
from backend.app.schemas.telemetry import (
    StoppedVehicleEventSchema,
    CongestionStateSchema,
    CongestionLevelEnum,
    TrafficAnalyticsSummarySchema,
    DetectionSummarySchema,
    TrackingSummarySchema,
)
from backend.app.services.vlm.keyframe_extractor import (
    KeyframeExtractor,
    ExtractedFrame,
    VideoReadError,
)
from backend.app.services.vlm.evidence_packager import EvidencePackager


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Creates a 50-frame, 25 fps, 640x360 synthetic MP4 video (2.0 seconds)."""
    video_file = tmp_path / "test_synth_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    width, height = 640, 360
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for f in range(50):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw moving rectangle
        x = int(50 + f * 5)
        cv2.rectangle(frame, (x, 100), (x + 80, 180), (0, 200, 255), -1)
        cv2.putText(frame, f"F:{f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return video_file


@pytest.fixture
def hd_synthetic_video_path(tmp_path: Path) -> Path:
    """Creates a 1920x1080 synthetic video to test aspect-ratio preserving downscaling."""
    video_file = tmp_path / "test_hd_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    width, height = 1920, 1080
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for f in range(10):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_file


# ============================================================================
# 1. Determinism and Budget Enforcement Tests
# ============================================================================

def test_deterministic_extraction_from_identical_input(synthetic_video_path: Path):
    """Verifies that running extraction twice with identical inputs produces identical results."""
    extractor = KeyframeExtractor()
    events = [
        StoppedVehicleEventSchema(
            track_id=1,
            start_frame=10,
            start_timestamp_s=0.40,
            end_frame=30,
            end_timestamp_s=1.20,
            duration_s=0.80,
            location=(100.0, 150.0),
            status="resolved",
        )
    ]

    frames1, diag1 = extractor.extract_keyframes(synthetic_video_path, stopped_events=events)
    frames2, diag2 = extractor.extract_keyframes(synthetic_video_path, stopped_events=events)

    assert len(frames1) == len(frames2)
    for f1, f2 in zip(frames1, frames2):
        assert f1.frame_idx == f2.frame_idx
        assert f1.timestamp_s == f2.timestamp_s
        assert f1.selection_reason == f2.selection_reason
        assert f1.associated_event_ids == f2.associated_event_ids
        assert len(f1.image_bytes) == len(f2.image_bytes)
        assert f1.width == f2.width
        assert f1.height == f2.height

    assert diag1.extracted_frame_count == diag2.extracted_frame_count
    assert diag1.total_candidates_identified == diag2.total_candidates_identified


def test_maximum_frame_budget_enforcement(synthetic_video_path: Path):
    """Verifies that extraction strictly enforces K=5 budget when many events exist."""
    extractor = KeyframeExtractor()
    config = ExtractionConfigSchema(frame_budget=5)

    # 8 distinct stopped-vehicle events
    events = [
        StoppedVehicleEventSchema(
            track_id=i,
            start_frame=i * 5,
            start_timestamp_s=i * 0.2,
            duration_s=float(i + 1),
            location=(50.0, 50.0),
            status="active",
        )
        for i in range(8)
    ]

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        config=config,
        stopped_events=events,
    )

    assert len(frames) == 5
    assert diagnostics.extracted_frame_count == 5
    assert diagnostics.candidates_pruned_budget > 0
    # Selected frames must be in chronological order
    frame_indices = [f.frame_idx for f in frames]
    assert frame_indices == sorted(frame_indices)


def test_fewer_candidates_than_budget(synthetic_video_path: Path):
    """Verifies behavior when available candidates are fewer than requested budget."""
    extractor = KeyframeExtractor()
    config = ExtractionConfigSchema(frame_budget=10)

    # 1 stopped vehicle event
    events = [
        StoppedVehicleEventSchema(
            track_id=1,
            start_frame=15,
            start_timestamp_s=0.60,
            duration_s=1.5,
            location=(100.0, 100.0),
            status="active",
        )
    ]

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        config=config,
        stopped_events=events,
    )

    # Should extract event frame plus periodic anchors without exceeding total available
    assert len(frames) <= 10
    assert len(frames) >= 1
    assert any("stopped_vehicle" in f.selection_reason for f in frames)
    assert any("scene_anchor" in f.selection_reason for f in frames)


# ============================================================================
# 2. Priority, Tie-Breaking, and Duplicate Suppression Tests
# ============================================================================

def test_stopped_vehicle_priority_over_anchors(synthetic_video_path: Path):
    """Verifies that high-importance events beat periodic anchors when budget is tight."""
    extractor = KeyframeExtractor()
    config = ExtractionConfigSchema(frame_budget=2)

    events = [
        StoppedVehicleEventSchema(
            track_id=101,
            start_frame=20,
            start_timestamp_s=0.80,
            duration_s=10.0,
            location=(200.0, 200.0),
            status="active",
        ),
        StoppedVehicleEventSchema(
            track_id=102,
            start_frame=35,
            start_timestamp_s=1.40,
            duration_s=8.0,
            location=(250.0, 250.0),
            status="active",
        ),
    ]

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        config=config,
        stopped_events=events,
    )

    assert len(frames) == 2
    # Both selected frames must be the stopped-vehicle events (priority 1), not periodic anchors (priority 3)
    reasons = [f.selection_reason for f in frames]
    assert any("stopped_vehicle_start_track_101" in r for r in reasons)
    assert any("stopped_vehicle_start_track_102" in r for r in reasons)


def test_duplicate_timestamp_suppression_same_event(synthetic_video_path: Path):
    """Verifies that near-duplicate frames for the same event or anchor are suppressed."""
    extractor = KeyframeExtractor()
    # Require 1.0s minimum interval
    config = ExtractionConfigSchema(frame_budget=5, min_event_interval_s=1.0)

    # Event with start and resolution only 0.2s apart
    events = [
        StoppedVehicleEventSchema(
            track_id=1,
            start_frame=10,
            start_timestamp_s=0.40,
            end_frame=15,
            end_timestamp_s=0.60,
            duration_s=0.20,
            location=(100.0, 100.0),
            status="resolved",
        )
    ]

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        config=config,
        stopped_events=events,
    )

    # The 0.60s resolution frame should be suppressed in favor of the 0.40s start frame
    assert diagnostics.candidates_pruned_duplicate > 0
    start_frames = [f for f in frames if "stopped_vehicle_start" in f.selection_reason]
    assert len(start_frames) == 1
    assert start_frames[0].frame_idx == 10


def test_distinct_events_within_short_interval_coexist_if_budget_allows(synthetic_video_path: Path):
    """Verifies that two distinct vehicle events occurring near in time are both preserved."""
    extractor = KeyframeExtractor()
    config = ExtractionConfigSchema(frame_budget=5, min_event_interval_s=1.0)

    events = [
        StoppedVehicleEventSchema(
            track_id=1,
            start_frame=10,
            start_timestamp_s=0.40,
            duration_s=5.0,
            location=(100.0, 100.0),
            status="active",
        ),
        StoppedVehicleEventSchema(
            track_id=2,  # Distinct track
            start_frame=15,
            start_timestamp_s=0.60,
            duration_s=4.0,
            location=(300.0, 100.0),
            status="active",
        ),
    ]

    frames, _ = extractor.extract_keyframes(
        synthetic_video_path,
        config=config,
        stopped_events=events,
    )

    # Both distinct events should be retained despite being 0.20s apart
    reasons = [f.selection_reason for f in frames]
    assert any("stopped_vehicle_start_track_1" in r for r in reasons)
    assert any("stopped_vehicle_start_track_2" in r for r in reasons)


# ============================================================================
# 3. Robustness, Clamping, and Error Handling Tests
# ============================================================================

def test_invalid_and_out_of_range_timestamps(synthetic_video_path: Path):
    """Verifies that out-of-range event timestamps are safely clamped to valid frame indices."""
    extractor = KeyframeExtractor()

    events = [
        StoppedVehicleEventSchema(
            track_id=99,
            start_frame=9999,
            start_timestamp_s=500.0,  # Far beyond 2.0s duration
            duration_s=5.0,
            location=(0.0, 0.0),
            status="active",
        )
    ]

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        stopped_events=events,
    )

    assert len(frames) > 0
    # All frame indices must be within [0, 49]
    for f in frames:
        assert 0 <= f.frame_idx < 50


def test_missing_fps_with_no_metadata_fps_timestamp_remains_null(synthetic_video_path: Path):
    """
    Verifies that when stream FPS is 0.0 or unavailable and no caller metadata FPS is provided:
    - Default 25 FPS fallback is NOT assumed.
    - diagnostics.source_fps is None.
    - diagnostics.source_duration_s is None.
    - Explicit diagnostic warning is logged.
    - Keyframe timestamps remain None.
    - Frame indices remain preserved and valid.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", fake_get):
        frames, diag = extractor.extract_keyframes(
            synthetic_video_path,
            source_fps=None,
        )

    assert diag.source_fps is None
    assert diag.source_duration_s is None
    assert any("unavailable or invalid" in w for w in diag.warnings)
    assert len(frames) > 0
    for f in frames:
        assert f.timestamp_s is None
        assert 0 <= f.frame_idx < 50


def test_invalid_fps_with_no_metadata_fps_timestamp_remains_null(synthetic_video_path: Path):
    """Verifies that NaN or negative FPS with invalid source_fps results in null timestamps."""
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return float("nan")
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", fake_get):
        frames, diag = extractor.extract_keyframes(
            synthetic_video_path,
            source_fps=0.0,  # Invalid metadata source_fps
        )

    assert diag.source_fps is None
    assert diag.source_duration_s is None
    assert len(frames) > 0
    for f in frames:
        assert f.timestamp_s is None


def test_valid_metadata_fps_fallback_calculates_correct_timestamp(synthetic_video_path: Path):
    """Verifies that valid caller-supplied metadata FPS is used when stream FPS is unavailable."""
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", fake_get):
        frames, diag = extractor.extract_keyframes(
            synthetic_video_path,
            source_fps=30.0,
        )

    assert diag.source_fps == 30.0
    assert any("using metadata source FPS: 30.00" in w for w in diag.warnings)
    assert len(frames) > 0
    for f in frames:
        assert f.timestamp_s is not None
        expected_ts = round(f.frame_idx / 30.0, 3)
        assert math.isclose(f.timestamp_s, expected_ts, abs_tol=1e-3)


def test_event_timestamp_cannot_be_associated_with_frame_whose_timestamp_is_unknown(synthetic_video_path: Path):
    """
    Verifies that when FPS is unavailable, an event specifying only start_timestamp_s
    without start_frame is NOT mapped to a frame index using an assumed FPS default.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return orig_get(self, prop)

    # Event with timestamp only and no frame index
    events = [
        {
            "track_id": 99,
            "start_timestamp_s": 1.0,
            "duration_s": 3.0,
            "start_frame": None,
        }
    ]

    with patch.object(cv2.VideoCapture, "get", fake_get):
        frames, diag = extractor.extract_keyframes(
            synthetic_video_path,
            source_fps=None,
            stopped_events=events,
        )

    # Frame 25 (1.0s * 25fps) must NOT be extracted for this event
    reasons = [f.selection_reason for f in frames]
    assert not any("track_99" in r for r in reasons)
    assert any("Assumed FPS mapping is prohibited" in w for w in diag.warnings)


def test_frame_index_remains_correct_when_timestamp_is_unknown(synthetic_video_path: Path):
    """
    Verifies that when FPS is unavailable, an event with an explicit start_frame
    preserves its frame index faithfully while setting timestamp_s to None.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return orig_get(self, prop)

    events = [
        StoppedVehicleEventSchema(
            track_id=42,
            start_frame=15,
            start_timestamp_s=0.60,
            duration_s=4.0,
            location=(100.0, 100.0),
            status="active",
        )
    ]

    with patch.object(cv2.VideoCapture, "get", fake_get):
        frames, diag = extractor.extract_keyframes(
            synthetic_video_path,
            source_fps=None,
            stopped_events=events,
        )

    matching = [f for f in frames if f.frame_idx == 15]
    assert len(matching) == 1
    assert matching[0].timestamp_s is None
    assert "stopped_vehicle_start_track_42" in matching[0].selection_reason


def test_frame_decoding_failure_releases_videocapture_resource(synthetic_video_path: Path):
    """Verifies that VideoCapture.release() is guaranteed to execute even if frame decode fails."""
    extractor = KeyframeExtractor()
    release_called = []
    orig_release = cv2.VideoCapture.release

    def tracking_release(self):
        release_called.append(True)
        return orig_release(self)

    # Force decode failure on frame read
    with patch.object(cv2.VideoCapture, "read", return_value=(False, None)):
        with patch.object(cv2.VideoCapture, "release", tracking_release):
            frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert len(release_called) > 0
    assert len(frames) == 0
    assert any("failed to decode frame" in w for w in diag.warnings)


def test_missing_event_timestamps_do_not_create_fabricated_timestamps():
    """
    Verifies that EvidencePackager preserves null event timestamps and does NOT fabricate
    a '0.00' string or float timestamp when timing is absent.
    """
    raw_event = {
        "track_id": 5,
        "start_timestamp_s": None,
        "end_timestamp_s": None,
        "start_frame": 20,
        "duration_s": None,
        "status": "active",
    }

    dummy_frame = ExtractedFrame(
        frame_idx=20,
        timestamp_s=None,
        selection_reason="test",
        associated_event_ids=["stop_5_f20"],
        image_bytes=b"dummy",
        image_base64="dummyb64",
        width=320,
        height=240,
    )

    package = EvidencePackager.package_evidence(
        video_id="v_null",
        job_id="j_null",
        sequence_name="s_null",
        extracted_keyframes=[dummy_frame],
        stopped_events=[raw_event],
    )

    assert len(package.events) == 1
    evt = package.events[0]
    assert evt.timestamp_s is None
    assert evt.duration_s is None
    assert evt.details["end_timestamp_s"] is None
    assert "0.00" not in evt.event_id
    assert package.keyframes[0].timestamp_s is None


def test_public_evidence_serialization_correctly_handles_null_timestamps():
    """
    Verifies that EvidencePackageSchema and EvidencePackagePublicSchema serialize
    nullable timestamps cleanly as null and deserialize back without validation errors.
    """
    dummy_frame = ExtractedFrame(
        frame_idx=10,
        timestamp_s=None,
        selection_reason="scene_anchor_frame_10",
        associated_event_ids=[],
        image_bytes=b"img",
        image_base64="aW1n",
        width=640,
        height=360,
    )

    package = EvidencePackager.package_evidence(
        video_id="v_serial",
        job_id="j_serial",
        sequence_name="s_serial",
        extracted_keyframes=[dummy_frame],
        stopped_events=[{"track_id": 1, "start_timestamp_s": None, "start_frame": 10}],
    )

    # Convert to public schema
    public_pkg = package.to_public_schema()
    json_str = public_pkg.model_dump_json()

    # JSON should represent null timestamps as literal null
    assert '"timestamp_s":null' in json_str.replace(" ", "")

    # Deserialize back
    restored = EvidencePackagePublicSchema.model_validate_json(json_str)
    assert restored.keyframes[0].timestamp_s is None
    assert restored.events[0].timestamp_s is None
    assert restored.keyframes[0].frame_idx == 10


def test_seek_skew_divergence_records_actual_decoded_frame(synthetic_video_path: Path):
    """
    Verifies that if decoder reports a position differing from the requested target frame,
    the actual decoded index is recorded, is_verified is True, and a diagnostic warning is emitted.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    # Simulate seek divergence: when querying CAP_PROP_POS_FRAMES after read, report frame + 4
    # (e.g. if next frame is 15.0, decoded frame was 14.0)
    def skewed_get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return 15.0  # (15 - 1 = 14)
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", skewed_get):
        frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert any("Seek position divergence on compressed container" in w for w in diag.warnings)
    assert len(frames) > 0
    for f in frames:
        assert f.frame_idx == 14
        assert f.requested_frame_idx is not None
        assert f.is_verified is True
    assert diag.verified_frame_count == len(frames)
    assert diag.unverified_frame_count == 0


def test_opencv_position_semantics_on_real_video(tmp_path: Path):
    """
    Controlled test establishing active OpenCV backend position semantics on actual video:
    - Verifies that querying CAP_PROP_POS_FRAMES after reading frame n returns n + 1.
    - Verifies that the decoded frame index is int(round(pos_after)) - 1.
    - Confirms that reading frame n does not produce an off-by-one recorded index.
    """
    video_path = tmp_path / "controlled_semantics.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    w, h = 320, 240
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    num_frames = 10
    for i in range(num_frames):
        img = np.full((h, w, 3), i * 20, dtype=np.uint8)
        writer.write(img)
    writer.release()

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    # 1. Sequential decoding verification
    for expected_idx in range(5):
        pos_before = cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert pos_before == float(expected_idx)

        ret, frame = cap.read()
        assert ret is True
        assert frame is not None

        pos_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # CAP_PROP_POS_FRAMES represents index of NEXT frame to decode (n + 1)
        assert pos_after == float(expected_idx + 1)

        # Derived index matches expected_idx without off-by-one error
        derived_idx = int(round(pos_after)) - 1
        assert derived_idx == expected_idx

    # 2. Seeking verification to known frames
    for target_idx in [0, 2, 6, 9]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        assert ret is True

        pos_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert pos_after == float(target_idx + 1)

        derived_idx = int(round(pos_after)) - 1
        assert derived_idx == target_idx

    cap.release()


def test_sequential_and_seeking_content_correspondence_on_real_video(tmp_path: Path):
    """
    Validates that KeyframeExtractor extracts the exact visual content of requested frames
    from a real video and marks them verified with matching frame indices.
    """
    video_path = tmp_path / "content_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    w, h = 160, 120
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    # Frame k has distinct solid pixel values (k * 25)
    for k in range(10):
        img = np.full((h, w, 3), k * 25, dtype=np.uint8)
        writer.write(img)
    writer.release()

    extractor = KeyframeExtractor()
    events = [
        StoppedVehicleEventSchema(
            track_id=1,
            start_frame=3,
            start_timestamp_s=0.12,
            duration_s=2.0,
            location=(50.0, 50.0),
            status="active",
        ),
        StoppedVehicleEventSchema(
            track_id=2,
            start_frame=7,
            start_timestamp_s=0.28,
            duration_s=2.0,
            location=(80.0, 80.0),
            status="active",
        ),
    ]

    frames, diag = extractor.extract_keyframes(
        video_path,
        stopped_events=events,
    )

    assert len(frames) > 0
    assert diag.verified_frame_count == len(frames)
    assert diag.unverified_frame_count == 0

    frame_3 = next((f for f in frames if f.requested_frame_idx == 3), None)
    assert frame_3 is not None
    assert frame_3.frame_idx == 3
    assert frame_3.is_verified is True

    # Decode JPEG bytes and check that image content matches frame 3 (3 * 25 = 75 +/- compression delta)
    decoded_img = cv2.imdecode(np.frombuffer(frame_3.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img is not None
    assert math.isclose(decoded_img[0, 0, 0], 3 * 25, abs_tol=15)

    frame_7 = next((f for f in frames if f.requested_frame_idx == 7), None)
    assert frame_7 is not None
    assert frame_7.frame_idx == 7
    assert frame_7.is_verified is True
    decoded_img_7 = cv2.imdecode(np.frombuffer(frame_7.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert math.isclose(decoded_img_7[0, 0, 0], 7 * 25, abs_tol=15)


def test_unavailable_decoder_position_marks_frame_unverified(synthetic_video_path: Path):
    """
    Verifies that when decoder position query returns -1.0 (unsupported), the frame
    is NOT falsely labeled as verified; distinction is preserved and warning emitted.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def unsupported_get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return -1.0
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", unsupported_get):
        frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert len(frames) > 0
    assert diag.unverified_frame_count == len(frames)
    assert diag.verified_frame_count == 0
    for f in frames:
        assert f.is_verified is False
        assert f.requested_frame_idx is not None
        assert f.frame_idx == f.requested_frame_idx

    assert any("Decoder position query unavailable or invalid" in w for w in diag.warnings)


def test_unadvancing_decoder_position_marks_frame_unverified(synthetic_video_path: Path):
    """
    Verifies that when CAP_PROP_POS_FRAMES returns 0.0 without advancing after read,
    the frame is flagged as unverified and an explicit diagnostic warning is emitted.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def static_zero_get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return 0.0
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", static_zero_get):
        frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert len(frames) > 0
    assert diag.unverified_frame_count == len(frames)
    for f in frames:
        assert f.is_verified is False
    assert any("Decoder position did not advance after read" in w for w in diag.warnings)


def test_inconsistent_out_of_range_position_marks_frame_unverified(synthetic_video_path: Path):
    """
    Verifies that an out-of-range post-read position report is caught, flagged as unverified,
    and does not produce a corrupted frame index.
    """
    extractor = KeyframeExtractor()
    orig_get = cv2.VideoCapture.get

    def corrupt_get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return 9999.0  # Far beyond video frame count (50)
        return orig_get(self, prop)

    with patch.object(cv2.VideoCapture, "get", corrupt_get):
        frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert len(frames) > 0
    assert diag.unverified_frame_count == len(frames)
    for f in frames:
        assert f.is_verified is False
    assert any("out-of-range post-read position" in w for w in diag.warnings)


def test_decoder_resource_released_on_success(synthetic_video_path: Path):
    """Verifies that VideoCapture.release() is guaranteed to execute on successful extraction."""
    extractor = KeyframeExtractor()
    release_called = []
    orig_release = cv2.VideoCapture.release

    def tracking_release(self):
        release_called.append(True)
        return orig_release(self)

    with patch.object(cv2.VideoCapture, "release", tracking_release):
        frames, diag = extractor.extract_keyframes(synthetic_video_path)

    assert len(release_called) == 1
    assert len(frames) > 0


def test_empty_video_raises_video_read_error(tmp_path: Path):
    """Verifies that an empty (0 bytes) file raises VideoReadError."""
    empty_file = tmp_path / "empty_video.mp4"
    empty_file.write_bytes(b"")

    extractor = KeyframeExtractor()
    with pytest.raises(VideoReadError, match="empty"):
        extractor.extract_keyframes(empty_file)


def test_nonexistent_video_raises_file_not_found(tmp_path: Path):
    """Verifies that a non-existent path raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.mp4"
    extractor = KeyframeExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_keyframes(missing_file)


def test_corrupt_video_raises_video_read_error(tmp_path: Path):
    """Verifies that invalid/corrupt non-video data raises VideoReadError."""
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_bytes(b"THIS IS NOT A VALID MP4 CONTAINER HEADER")

    extractor = KeyframeExtractor()
    with pytest.raises(VideoReadError):
        extractor.extract_keyframes(corrupt_file)


# ============================================================================
# 4. Image Resizing and Aspect Ratio Tests
# ============================================================================

def test_aspect_ratio_preserving_downscaling(hd_synthetic_video_path: Path):
    """Verifies that 1920x1080 frames are scaled to 960x540 maintaining 16:9 ratio."""
    extractor = KeyframeExtractor()
    config = ExtractionConfigSchema(max_dimension=960, frame_budget=2)

    frames, _ = extractor.extract_keyframes(hd_synthetic_video_path, config=config)

    assert len(frames) > 0
    for f in frames:
        assert f.width == 960
        assert f.height == 540
        # Aspect ratio check: 960/540 == 1920/1080 (16/9)
        assert math.isclose(f.width / f.height, 16.0 / 9.0, rel_tol=1e-2)
        # Check image can be decoded back
        decoded = cv2.imdecode(np.frombuffer(f.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[1] == 960
        assert decoded.shape[0] == 540


# ============================================================================
# 5. EvidencePackager Integration & Contract Isolation Tests
# ============================================================================

def test_evidence_packager_assembly_and_public_isolation(synthetic_video_path: Path):
    """
    Verifies that EvidencePackager builds complete internal packages and that
    to_public_schema() strictly isolates binary image data and filesystem paths.
    """
    extractor = KeyframeExtractor()
    events = [
        StoppedVehicleEventSchema(
            track_id=7,
            start_frame=12,
            start_timestamp_s=0.48,
            duration_s=6.5,
            location=(120.0, 180.0),
            roi_id="roi_test",
            status="active",
        )
    ]

    analytics_summary = TrafficAnalyticsSummarySchema(
        sequence_name="seq_synth_01",
        total_frames=50,
        duration_s=2.0,
        fps=25.0,
        total_unique_vehicles=3,
        line_counts={"line_1": {"downstream": 2}},
        stopped_vehicle_events=events,
        peak_occupancy_per_roi={"roi_test": 0.25},
        average_occupancy_per_roi={"roi_test": 0.12},
    )

    det_summary = DetectionSummarySchema(
        total_detections=150,
        mean_confidence=0.89,
        class_distribution={"vehicle": 150},
    )

    track_summary = TrackingSummarySchema(
        total_unique_tracks=3,
    )

    frames, diagnostics = extractor.extract_keyframes(
        synthetic_video_path,
        stopped_events=events,
    )

    # Package internal evidence
    internal_package = EvidencePackager.package_evidence(
        video_id="vid_12345",
        job_id="job_67890",
        sequence_name="seq_synth_01",
        extracted_keyframes=frames,
        analytics_summary=analytics_summary,
        detection_summary=det_summary,
        tracking_summary=track_summary,
        stopped_events=events,
        diagnostics=diagnostics,
    )

    assert isinstance(internal_package, EvidencePackageSchema)
    assert internal_package.video_id == "vid_12345"
    assert internal_package.job_id == "job_67890"
    assert len(internal_package.keyframes) == len(frames)
    assert len(internal_package.events) == 1
    assert internal_package.events[0].track_id == 7
    assert internal_package.events[0].duration_s == 6.5
    assert internal_package.telemetry_summary["total_unique_tracks"] == 3
    assert internal_package.telemetry_summary["total_detections"] == 150

    # Verify internal schema contains base64 image data
    for kf in internal_package.keyframes:
        assert len(kf.image_base64) > 100
        assert kf.content_type == "image/jpeg"

    # Convert to Public Schema
    public_package = internal_package.to_public_schema()
    assert isinstance(public_package, EvidencePackagePublicSchema)
    assert public_package.job_id == "job_67890"
    assert len(public_package.keyframes) == len(frames)

    # Verify public schema has NO base64 image payload
    public_dict = public_package.model_dump()
    for kf_dict in public_dict["keyframes"]:
        assert "image_base64" not in kf_dict
        assert "width" in kf_dict
        assert "height" in kf_dict
        assert "timestamp_s" in kf_dict
        assert "frame_idx" in kf_dict

    # Verify public schema contains NO local filesystem storage paths
    serialized_json = public_package.model_dump_json()
    assert "C:\\" not in serialized_json
    assert "/storage/" not in serialized_json
    assert "\\storage\\" not in serialized_json


def test_evidence_packager_handles_missing_telemetry_gracefully(synthetic_video_path: Path):
    """Verifies that EvidencePackager gracefully handles None or empty summaries."""
    extractor = KeyframeExtractor()
    frames, diagnostics = extractor.extract_keyframes(synthetic_video_path)

    package = EvidencePackager.package_evidence(
        video_id="vid_bare",
        job_id="job_bare",
        sequence_name="seq_bare",
        extracted_keyframes=frames,
        analytics_summary=None,
        detection_summary=None,
        tracking_summary=None,
        diagnostics=diagnostics,
    )

    assert package.job_id == "job_bare"
    assert package.events == []
    assert package.telemetry_summary == {}
    assert len(package.keyframes) == len(frames)
