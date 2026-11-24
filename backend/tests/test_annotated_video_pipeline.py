"""
Comprehensive regression tests for annotated video generation, codec fallback,
odd dimension handling, verification, failure isolation, and diagnostics.
"""

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from backend.app.services.cv_pipeline_adapter import CVPipelineAdapter
from backend.app.schemas import (
    DetectionSummarySchema,
    TrackingSummarySchema,
    TrafficAnalyticsSummarySchema,
)


def _create_synthetic_video(
    path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 25.0,
    num_frames: int = 10,
    codec: str = "mp4v",
) -> Path:
    """Helper creating a playable synthetic test video with moving vehicle-like boxes."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        # Draw roadway lane lines
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        # Draw moving rectangle
        x = int((i / num_frames) * (width - 60))
        y = height // 4
        cv2.rectangle(frame, (x, y), (x + 50, y + 30), (0, 165, 255), -1)
        vw.write(frame)
    vw.release()
    return path


def test_supported_mp4_codec(tmp_path):
    """Verifies that standard MP4 produces verified, playable annotated video with avc1/H264."""
    video_path = _create_synthetic_video(tmp_path / "input.mp4", width=640, height=480, fps=25.0)
    out_dir = tmp_path / "out"

    report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
        video_path=video_path,
        output_dir=out_dir,
        sequence_name="std_test",
        generate_annotated_video=True,
    )

    assert "annotated_video" in artifacts
    assert Path(artifacts["annotated_video"]).exists()
    assert Path(artifacts["annotated_video"]).stat().st_size > 0
    assert diag["status"] == "COMPLETED"
    assert diag["verification_passed"] is True
    assert diag["frames_written"] == 10
    assert diag["selected_codec"] in ("avc1", "H264", "X264", "mp4v")


def test_non_ascii_unicode_sequence_name(tmp_path):
    """Verifies that sequence names with Unicode characters (ellipsis '…') do not cause file mangling."""
    video_path = _create_synthetic_video(tmp_path / "input.mp4", width=640, height=480, fps=24.0)
    out_dir = tmp_path / "out_unicode"

    unicode_seq = "Vehicle_stalling_causing_traffic…_e358e42d"
    report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
        video_path=video_path,
        output_dir=out_dir,
        sequence_name=unicode_seq,
        generate_annotated_video=True,
    )

    assert "annotated_video" in artifacts
    ann_path = Path(artifacts["annotated_video"])
    assert ann_path.exists()
    assert ann_path.stat().st_size > 0
    assert diag["status"] == "COMPLETED"
    assert diag["verification_passed"] is True
    # Ensure filename is pure ASCII safe
    assert "…" not in ann_path.name


def test_different_resolutions(tmp_path):
    """Verifies pipeline across multiple resolutions (160x120, 640x480, 960x540, 1280x720)."""
    resolutions = [(160, 120), (640, 480), (960, 540)]
    for w, h in resolutions:
        vpath = _create_synthetic_video(tmp_path / f"in_{w}x{h}.mp4", width=w, height=h, num_frames=5)
        out_dir = tmp_path / f"out_{w}x{h}"

        report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
            video_path=vpath,
            output_dir=out_dir,
            sequence_name=f"seq_{w}x{h}",
        )

        assert "annotated_video" in artifacts
        assert diag["verification_passed"] is True
        assert diag["output_metadata"]["width"] == w
        assert diag["output_metadata"]["height"] == h


def test_different_fps_values(tmp_path):
    """Verifies video writing across various framerates (15.0, 24.0, 30.0)."""
    for fps in (15.0, 24.0, 30.0):
        vpath = _create_synthetic_video(tmp_path / f"in_{fps}fps.mp4", width=320, height=240, fps=fps, num_frames=8)
        out_dir = tmp_path / f"out_{fps}fps"

        report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
            video_path=vpath,
            output_dir=out_dir,
            sequence_name=f"seq_{fps}fps",
        )

        assert "annotated_video" in artifacts
        assert diag["verification_passed"] is True
        assert diag["output_metadata"]["fps"] == fps


def test_odd_frame_dimensions(tmp_path):
    """Verifies odd dimensions (e.g. 641x481) are adjusted to even dimensions and encoded cleanly."""
    odd_w, odd_h = 641, 481
    vpath = _create_synthetic_video(tmp_path / "odd_input.mp4", width=odd_w, height=odd_h, num_frames=6)
    out_dir = tmp_path / "out_odd"

    report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
        video_path=vpath,
        output_dir=out_dir,
        sequence_name="odd_dims",
    )

    assert "annotated_video" in artifacts
    assert diag["verification_passed"] is True
    # Target dimensions must be even for H.264 compatibility
    assert diag["output_metadata"]["width"] == 640
    assert diag["output_metadata"]["height"] == 480

    # Ensure output can be reopened and read
    cap = cv2.VideoCapture(artifacts["annotated_video"])
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret is True
    assert frame.shape == (480, 640, 3)
    cap.release()


def test_videowriter_init_failure_preserves_analytics(tmp_path, monkeypatch):
    """
    CRITICAL ISOLATION TEST: When VideoWriter initialization fails for all codecs,
    the pipeline MUST NOT raise an unhandled exception. It must produce full CV analytics
    and record annotation failure diagnostics.
    """
    video_path = _create_synthetic_video(tmp_path / "input.mp4", width=640, height=480, num_frames=5)
    out_dir = tmp_path / "out_fail"

    # Mock cv2.VideoWriter to fail opening
    real_video_writer = cv2.VideoWriter

    class FailingVideoWriter:
        def __init__(self, *args, **kwargs):
            pass
        def isOpened(self):
            return False
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", FailingVideoWriter)

    report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
        video_path=video_path,
        output_dir=out_dir,
        sequence_name="init_fail_test",
        generate_annotated_video=True,
    )

    # Core CV analytics MUST be preserved
    assert report.total_frames == 5
    assert det.total_detections >= 0
    assert "analytics_json" in artifacts
    assert "telemetry_csv" in artifacts

    # Annotated video must NOT be registered
    assert "annotated_video" not in artifacts
    assert diag["status"] == "FAILED"
    assert diag["writer_initialized"] is False
    assert "Failed to initialize VideoWriter" in diag["error_message"]


def test_corrupt_or_zero_byte_output_file(tmp_path, monkeypatch):
    """Verifies that a corrupt 0-byte output file fails verification, is deleted, and not registered."""
    video_path = _create_synthetic_video(tmp_path / "input.mp4", width=640, height=480, num_frames=5)
    out_dir = tmp_path / "out_corrupt"

    # Mock VideoWriter to write nothing and create an empty file
    class ZeroByteVideoWriter:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)
            self.path.touch()  # 0-byte file
        def isOpened(self):
            return True
        def write(self, frame):
            pass  # Does not write any bytes
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", ZeroByteVideoWriter)

    report, det, track, artifacts, diag = CVPipelineAdapter.process_video_file(
        video_path=video_path,
        output_dir=out_dir,
        sequence_name="corrupt_test",
        generate_annotated_video=True,
    )

    assert "annotated_video" not in artifacts
    assert diag["status"] == "FAILED"
    assert diag["verification_passed"] is False
    # Ensure 0-byte file was cleaned up
    assert not (out_dir / "corrupt_test_annotated.mp4").exists()
