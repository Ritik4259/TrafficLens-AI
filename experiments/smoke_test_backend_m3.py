"""
End-to-end real-video smoke test and performance benchmark for Backend Milestone 3.

Runs against the real FastAPI application, SQLAlchemy persistence, JobWorker,
and the actual CV pipeline (YOLO11n + ByteTrack + TrafficAnalyticsPipeline + OpenCV).
No mocking or monkeypatching.
"""

from __future__ import annotations
import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import psutil
import torch
from fastapi.testclient import TestClient

from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.schemas.job import JobStatus


def run_smoke_test():
    print("=" * 80)
    print("STARTING BACKEND MILESTONE 3 REAL-VIDEO SMOKE TEST")
    print("=" * 80)

    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    print(f"System State: Initial Memory={initial_memory_mb:.2f} MB | CUDA Available={cuda_available} (Inference on {device_name})")

    video_path = Path("storage/smoke_test_real_traffic.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found at {video_path}")

    video_bytes = video_path.read_bytes()
    file_size_bytes = len(video_bytes)
    print(f"Loaded Real Video Clip: {video_path} ({file_size_bytes / (1024*1024):.2f} MB)")

    # Run inside application lifespan (starts JobWorker and recovery routines)
    with TestClient(app) as client:
        # Clean up any leftover QUEUED jobs from previous test runs
        with SessionLocal() as db:
            from sqlalchemy import delete
            db.execute(delete(ProcessingJob).where(ProcessingJob.status == JobStatus.QUEUED))
            db.commit()

        print("\n--- 1. UPLOAD REAL VIDEO VIA POST /api/v1/videos ---")
        t_upload_start = time.perf_counter()
        upload_resp = client.post(
            "/api/v1/videos",
            files={"file": ("real_traffic_mvi_20011.mp4", io.BytesIO(video_bytes), "video/mp4")},
        )
        t_upload_elapsed = time.perf_counter() - t_upload_start

        assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
        upload_data = upload_resp.json()
        video_id = upload_data["video_id"]
        meta = upload_data["metadata"]
        print(f"Upload Status: HTTP 201 Created in {t_upload_elapsed*1000:.1f} ms")
        print(f"Generated Video ID: {video_id}")
        print(f"Extracted Metadata: {meta['width']}x{meta['height']} @ {meta['fps']} FPS, {meta['frame_count']} frames, {meta['duration_seconds']}s")

        # 2. Verify persisted VideoAsset in Database
        print("\n--- 2. VERIFY PERSISTED VIDEOASSET IN DATABASE ---")
        with SessionLocal() as db:
            asset = db.get(VideoAsset, video_id)
            assert asset is not None, f"VideoAsset {video_id} not found in database!"
            assert asset.file_size_bytes == file_size_bytes
            assert asset.width == meta["width"]
            assert asset.height == meta["height"]
            assert asset.fps == meta["fps"]
            assert Path(asset.storage_path).exists()
            print(f"Verified DB VideoAsset: ID={asset.id}, Storage Path={asset.storage_path} (File exists on disk: {Path(asset.storage_path).stat().st_size} bytes)")

        # 3. Create Processing Job via POST /api/v1/videos/{video_id}/jobs
        print("\n--- 3. CREATE PROCESSING JOB VIA POST /api/v1/videos/{video_id}/jobs ---")
        job_payload = {
            "sequence_name": "MVI_20011_SMOKE",
            "lines": [
                {
                    "line_id": "line_mid",
                    "name": "Middle Crossline",
                    "p1": [200.0, 360.0],
                    "p2": [800.0, 360.0],
                    "positive_direction_label": "downstream",
                    "negative_direction_label": "upstream",
                    "cooldown_frames": 10,
                    "min_track_age": 2,
                }
            ],
            "rois": [
                {
                    "roi_id": "roi_main_flow",
                    "name": "Roadway Surface",
                    "polygon": [
                        [250.0, 180.0],
                        [750.0, 180.0],
                        [920.0, 520.0],
                        [80.0, 520.0],
                    ],
                    "description": "Main roadway flow zone",
                }
            ],
        }

        job_resp = client.post(f"/api/v1/videos/{video_id}/jobs", json=job_payload)
        assert job_resp.status_code == 202, f"Job creation failed: {job_resp.text}"
        job_data = job_resp.json()
        job_id = job_data["job_id"]
        print(f"Job Creation Status: HTTP 202 Accepted")
        print(f"Job ID: {job_id} | Initial Status: {job_data['status']}")

        # 4. Observe Persisted Status Transitions
        print("\n--- 4. OBSERVE PERSISTED STATUS TRANSITIONS ---")
        observed_statuses = [job_data["status"]]
        t_proc_start = time.perf_counter()
        final_job_data = None
        max_wait_seconds = 60.0

        while time.perf_counter() - t_proc_start < max_wait_seconds:
            status_resp = client.get(f"/api/v1/jobs/{job_id}")
            assert status_resp.status_code == 200
            current_status = status_resp.json()["status"]

            if current_status != observed_statuses[-1]:
                observed_statuses.append(current_status)
                print(f"[{time.perf_counter() - t_proc_start:.2f}s] Status transitioned -> {current_status}")

            if current_status in ("COMPLETED", "FAILED"):
                final_job_data = status_resp.json()
                break

            time.sleep(0.3)

        t_proc_elapsed = time.perf_counter() - t_proc_start
        peak_memory_mb = process.memory_info().rss / (1024 * 1024)

        assert final_job_data is not None, "Job timed out before reaching a terminal status!"
        print(f"Observed Lifecycle Transitions: {' -> '.join(observed_statuses)}")
        assert final_job_data["status"] == "COMPLETED", f"Job failed: {final_job_data.get('error')}"
        print(f"Job Successfully COMPLETED in {t_proc_elapsed:.2f} seconds (reported duration: {final_job_data['duration_seconds']}s)")

        # 5. Retrieve Results via GET /api/v1/jobs/{job_id}/results
        print("\n--- 5. RETRIEVE RESULTS VIA GET /api/v1/jobs/{job_id}/results ---")
        results_resp = client.get(f"/api/v1/jobs/{job_id}/results")
        assert results_resp.status_code == 200, f"Results query failed: {results_resp.text}"
        res = results_resp.json()

        det = res["detection_summary"]
        trk = res["tracking_summary"]
        analytics = res["analytics_summary"]
        artifacts = res["artifacts"]

        print(f"Detection Summary: {det['total_detections']} detections across {meta['frame_count']} frames (mean confidence: {det['mean_confidence']:.3f})")
        print(f"Class Distribution: {det['class_distribution']}")
        print(f"Tracking Summary: {trk['total_unique_tracks']} unique tracks tracked across video")
        print(f"Analytics Summary: Total Unique Vehicles={analytics['total_unique_vehicles']}, Line Counts={analytics['line_counts']}")
        print(f"Peak ROI Occupancy: {analytics['peak_occupancy_per_roi']}")
        print(f"Congestion Distribution: {analytics['congestion_distribution_per_roi']}")
        print(f"Artifacts Generated: {artifacts}")

        # Assertions on analytics validity
        assert det["total_detections"] > 0, "No vehicles detected in real traffic frames!"
        assert trk["total_unique_tracks"] > 0, "No tracks generated in real traffic frames!"
        assert analytics["total_unique_vehicles"] > 0, "Analytics reported 0 unique vehicles!"

        # 6. Inspect Artifacts on Disk
        print("\n--- 6. INSPECT OUTPUT ARTIFACTS ON DISK ---")
        json_path = Path(artifacts["analytics_json"])
        csv_path = Path(artifacts["telemetry_csv"])
        video_out_path = Path(artifacts["annotated_video"])

        assert json_path.exists(), f"analytics.json missing at {json_path}"
        assert csv_path.exists(), f"telemetry.csv missing at {csv_path}"
        assert video_out_path.exists(), f"annotated.mp4 missing at {video_out_path}"

        # 6.1 Validate analytics.json
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)
        assert json_content["summary"]["sequence_name"] == "MVI_20011_SMOKE"
        assert len(json_content["frame_telemetries"]) == meta["frame_count"]
        print(f"[OK] analytics.json: Valid JSON, {json_path.stat().st_size} bytes, contains {len(json_content['frame_telemetries'])} frame telemetry entries.")

        # 6.2 Validate telemetry.csv
        csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        header = csv_lines[0]
        data_rows = len(csv_lines) - 1
        assert "frame_idx" in header and "timestamp_s" in header
        assert data_rows == meta["frame_count"]
        print(f"[OK] telemetry.csv: Valid CSV, {csv_path.stat().st_size} bytes, header=[{header[:60]}...], {data_rows} data rows.")

        # 6.3 Validate annotated.mp4 playability and geometry
        cap_out = cv2.VideoCapture(str(video_out_path))
        assert cap_out.isOpened(), "Annotated video cannot be opened by OpenCV!"
        out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = float(cap_out.get(cv2.CAP_PROP_FPS))
        out_frames = 0
        while True:
            ret, frame = cap_out.read()
            if not ret or frame is None:
                break
            out_frames += 1
        cap_out.release()

        assert out_w == meta["width"], f"Annotated video width mismatch: {out_w} vs {meta['width']}"
        assert out_h == meta["height"], f"Annotated video height mismatch: {out_h} vs {meta['height']}"
        assert out_frames == meta["frame_count"], f"Annotated video frame count mismatch: {out_frames} vs {meta['frame_count']}"
        print(f"[OK] annotated.mp4: Playable video stream, {video_out_path.stat().st_size} bytes, {out_w}x{out_h} @ {out_fps:.1f} FPS, {out_frames} decodable frames.")

    # 7. Summary Performance Metrics
    inference_fps = meta["frame_count"] / t_proc_elapsed
    print("\n" + "=" * 80)
    print("SMOKE TEST PERFORMANCE & RESOURCE REPORT")
    print("=" * 80)
    print(f"Video Clip Duration:        {meta['duration_seconds']:.2f} seconds")
    print(f"Video Frame Count:          {meta['frame_count']} frames (Resolution: {meta['width']}x{meta['height']})")
    print(f"End-to-End Processing Time: {t_proc_elapsed:.2f} seconds")
    print(f"Average Pipeline FPS:       {inference_fps:.2f} FPS")
    print(f"Initial Memory Usage:       {initial_memory_mb:.2f} MB")
    print(f"Peak Memory Usage:          {peak_memory_mb:.2f} MB (Delta: +{peak_memory_mb - initial_memory_mb:.2f} MB)")
    print(f"Inference Device:           {device_name.upper()} (PyTorch CUDA Available: {cuda_available})")
    print(f"Detections Count:           {det['total_detections']} vehicle bounding boxes")
    print(f"Unique Tracks Count:        {trk['total_unique_tracks']} vehicle IDs assigned")
    print(f"Smoke Test Result:          PASSED")
    print("=" * 80)

    # Save summary report as JSON for documentation
    summary_data = {
        "video_duration_s": meta["duration_seconds"],
        "video_frame_count": meta["frame_count"],
        "resolution": f"{meta['width']}x{meta['height']}",
        "processing_time_s": round(t_proc_elapsed, 2),
        "pipeline_fps": round(inference_fps, 2),
        "initial_memory_mb": round(initial_memory_mb, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "memory_delta_mb": round(peak_memory_mb - initial_memory_mb, 2),
        "inference_device": device_name.upper(),
        "cuda_available": cuda_available,
        "total_detections": det["total_detections"],
        "class_distribution": det["class_distribution"],
        "total_unique_tracks": trk["total_unique_tracks"],
        "total_unique_vehicles": analytics["total_unique_vehicles"],
        "artifacts": artifacts,
    }
    with open("storage/smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


if __name__ == "__main__":
    run_smoke_test()
