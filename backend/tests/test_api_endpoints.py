"""
Tests for FastAPI HTTP endpoints.
"""

from __future__ import annotations
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Validates the root service discovery endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TrafficLens-AI"
    assert "health_url" in data


def test_health_check_endpoint():
    """Validates /api/v1/health status and service metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TrafficLens-AI"
    assert data["database_dialect"] == "sqlite"
    assert data["database_connected"] is True


def test_list_videos_endpoint():
    """Validates GET /api/v1/videos returns list of video metadata."""
    response = client.get("/api/v1/videos")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_and_query_job_contract():
    """Validates job creation and query contracts."""
    import uuid
    test_video_id = f"video_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "video_id": test_video_id,
        "sequence_name": "MVI_20011",
        "lines": [
            {
                "line_id": "line_1",
                "name": "Main Crossline",
                "p1": [100.0, 200.0],
                "p2": [400.0, 200.0],
                "positive_direction_label": "downstream",
                "negative_direction_label": "upstream",
                "cooldown_frames": 15,
                "min_track_age": 2,
            }
        ],
    }

    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "QUEUED"
    assert data["video_id"] == test_video_id
    assert data["sequence_name"] == "MVI_20011"
    assert "job_id" in data

    # Query status endpoint
    job_id = data["job_id"]
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id
