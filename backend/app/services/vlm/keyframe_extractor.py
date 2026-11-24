"""
Deterministic keyframe extraction service for TrafficLens-AI.

Extracts representative video frames based on event triggers, congestion transitions,
and periodic scene anchors while enforcing temporal diversity and frame budgets.
"""

from __future__ import annotations
import base64
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np

from backend.app.core.logging import get_logger
from backend.app.schemas.vlm import ExtractionConfigSchema, ExtractionDiagnosticsSchema

logger = get_logger("trafficlens.vlm.extractor")


class VideoReadError(Exception):
    """Raised when a video file cannot be opened, is corrupt, or contains no readable frames."""
    pass


@dataclass
class CandidateFrame:
    """Represents a potential keyframe candidate before deduplication and budget pruning."""
    frame_idx: int
    timestamp_s: Optional[float]
    priority: int  # 1: primary stopped-vehicle, 2: peak congestion/occupancy, 3: periodic anchor, 4: pre-event context
    importance_score: float  # higher means higher priority within the same priority tier
    selection_reason: str
    associated_event_id: Optional[str] = None
    associated_track_id: Optional[int] = None


@dataclass
class ExtractedFrame:
    """In-memory representation of a decoded, resized, and encoded keyframe."""
    frame_idx: int
    timestamp_s: Optional[float]
    selection_reason: str
    associated_event_ids: List[str]
    image_bytes: bytes
    image_base64: str
    width: int
    height: int
    requested_frame_idx: Optional[int] = None
    is_verified: bool = True


class KeyframeExtractor:
    """
    Deterministic video keyframe extraction engine.
    
    Harvests candidate frames from spatiotemporal events, suppresses near-duplicates,
    enforces a hard frame budget (K), resizes images preserving aspect ratio,
    and returns encoded frames with associated telemetry timestamps.
    """

    def __init__(self, default_config: Optional[ExtractionConfigSchema] = None):
        self.config = default_config or ExtractionConfigSchema()

    def extract_keyframes(
        self,
        video_path: Union[str, Path],
        config: Optional[ExtractionConfigSchema] = None,
        stopped_events: Optional[List[Any]] = None,
        congestion_states: Optional[List[Any]] = None,
        peak_occupancy_per_roi: Optional[Dict[str, float]] = None,
        source_fps: Optional[float] = None,
    ) -> Tuple[List[ExtractedFrame], ExtractionDiagnosticsSchema]:
        """
        Extracts up to K keyframes from the given video file deterministically.

        Args:
            video_path: Filesystem path to the source video.
            config: Optional override configuration for extraction parameters.
            stopped_events: Optional list of StoppedVehicleEvent or StoppedVehicleEventSchema objects.
            congestion_states: Optional list of CongestionState or CongestionStateSchema objects.
            peak_occupancy_per_roi: Optional mapping of ROI ID to peak occupancy ratio.
            source_fps: Optional known source FPS from video metadata.

        Returns:
            Tuple of (extracted_frames_list, extraction_diagnostics).

        Raises:
            FileNotFoundError: If video_path does not exist on disk.
            VideoReadError: If video_path is empty (0 bytes) or cannot be opened by OpenCV.
        """
        active_config = config or self.config
        path = Path(video_path)
        diagnostics = ExtractionDiagnosticsSchema()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found at path: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise VideoReadError(f"Video file is empty (0 bytes): {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoReadError(f"OpenCV could not open video stream: {path}")

        try:
            # Inspect stream properties
            stream_fps = cap.get(cv2.CAP_PROP_FPS)
            stream_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if stream_fps is not None and math.isfinite(stream_fps) and stream_fps > 0:
                fps: Optional[float] = float(stream_fps)
            elif source_fps is not None and math.isfinite(source_fps) and source_fps > 0:
                fps = float(source_fps)
                diagnostics.warnings.append(
                    f"Stream FPS invalid or unavailable ({stream_fps}); using metadata source FPS: {fps:.2f}"
                )
            else:
                fps = None
                diagnostics.warnings.append(
                    f"Stream FPS ({stream_fps}) and metadata source FPS ({source_fps}) are unavailable or invalid. "
                    "Timestamps cannot be established reliably and will be set to None; frame indices are preserved."
                )

            diagnostics.source_fps = fps
            diagnostics.source_total_frames = stream_total_frames
            duration_s = (stream_total_frames / fps) if (fps is not None and fps > 0 and stream_total_frames > 0) else None
            diagnostics.source_duration_s = round(duration_s, 3) if duration_s is not None else None

            if stream_total_frames <= 0:
                diagnostics.warnings.append("Video stream reported 0 total frames; stream may be truncated or corrupt.")
                return [], diagnostics

            # 1. Harvest Candidates
            candidates = self._harvest_candidates(
                total_frames=stream_total_frames,
                fps=fps,
                duration_s=duration_s,
                config=active_config,
                stopped_events=stopped_events,
                congestion_states=congestion_states,
                peak_occupancy_per_roi=peak_occupancy_per_roi,
                diagnostics=diagnostics,
            )
            diagnostics.total_candidates_identified = len(candidates)

            # 2. Prune duplicates & enforce temporal separation
            pruned_candidates = self._suppress_duplicates(candidates, active_config, diagnostics)

            # 3. Enforce frame budget (K) with deterministic ranking
            selected_candidates = self._enforce_budget(pruned_candidates, active_config, diagnostics)

            # 4. Extract and resize frames from video
            extracted_frames = self._decode_and_encode_frames(
                cap=cap,
                candidates=selected_candidates,
                fps=fps,
                total_frames=stream_total_frames,
                config=active_config,
                diagnostics=diagnostics,
            )
            diagnostics.extracted_frame_count = len(extracted_frames)

            return extracted_frames, diagnostics

        finally:
            cap.release()

    def _harvest_candidates(
        self,
        total_frames: int,
        fps: Optional[float],
        duration_s: Optional[float],
        config: ExtractionConfigSchema,
        stopped_events: Optional[List[Any]],
        congestion_states: Optional[List[Any]],
        peak_occupancy_per_roi: Optional[Dict[str, float]],
        diagnostics: ExtractionDiagnosticsSchema,
    ) -> List[CandidateFrame]:
        """Harvests candidate keyframes from events, congestion spikes, and periodic anchors."""
        candidates: List[CandidateFrame] = []

        # A. Stopped Vehicle Events (Highest Priority = 1)
        if stopped_events:
            for idx, event in enumerate(stopped_events):
                # Accommodate both Pydantic models and dataclasses/dicts
                track_id = getattr(event, "track_id", None)
                if track_id is None and isinstance(event, dict):
                    track_id = event.get("track_id")

                start_ts = getattr(event, "start_timestamp_s", None)
                if start_ts is None and isinstance(event, dict):
                    start_ts = event.get("start_timestamp_s")

                end_ts = getattr(event, "end_timestamp_s", None)
                if end_ts is None and isinstance(event, dict):
                    end_ts = event.get("end_timestamp_s")

                start_frame = getattr(event, "start_frame", None)
                if start_frame is None and isinstance(event, dict):
                    start_frame = event.get("start_frame")

                end_frame = getattr(event, "end_frame", None)
                if end_frame is None and isinstance(event, dict):
                    end_frame = event.get("end_frame")

                duration = getattr(event, "duration_s", None)
                if duration is None and isinstance(event, dict):
                    duration = event.get("duration_s", 0.0)
                duration = float(duration or 0.0)

                # Case 1: Reliable FPS is available
                if fps is not None and fps > 0:
                    ts_str = f"{start_ts:.2f}" if (start_ts is not None and math.isfinite(start_ts)) else f"f{start_frame or idx}"
                    event_id = f"stop_{track_id or idx}_{ts_str}"

                    if start_ts is not None and math.isfinite(start_ts):
                        clamped_start_ts = max(0.0, min(duration_s, float(start_ts))) if duration_s is not None else max(0.0, float(start_ts))
                        cand_frame = int(round(clamped_start_ts * fps))
                        if start_frame is not None and 0 <= int(start_frame) < total_frames:
                            cand_frame = int(start_frame)
                        else:
                            cand_frame = max(0, min(total_frames - 1, cand_frame))

                        candidates.append(
                            CandidateFrame(
                                frame_idx=cand_frame,
                                timestamp_s=round(clamped_start_ts, 3),
                                priority=1,
                                importance_score=duration,
                                selection_reason=f"stopped_vehicle_start_track_{track_id}",
                                associated_event_id=event_id,
                                associated_track_id=track_id,
                            )
                        )

                        # Optional pre-event context frame
                        if config.include_pre_event_context and clamped_start_ts >= config.pre_event_offset_s:
                            pre_ts = max(0.0, clamped_start_ts - config.pre_event_offset_s)
                            pre_frame = max(0, min(total_frames - 1, int(round(pre_ts * fps))))
                            if pre_frame != cand_frame:
                                candidates.append(
                                    CandidateFrame(
                                        frame_idx=pre_frame,
                                        timestamp_s=round(pre_ts, 3),
                                        priority=4,  # Lower priority than primary event
                                        importance_score=duration * 0.5,
                                        selection_reason=f"stopped_vehicle_pre_context_track_{track_id}",
                                        associated_event_id=event_id,
                                        associated_track_id=track_id,
                                    )
                                )

                        # Event resolution frame
                        if end_ts is not None and math.isfinite(end_ts) and end_ts > start_ts:
                            clamped_end_ts = max(0.0, min(duration_s, float(end_ts))) if duration_s is not None else max(0.0, float(end_ts))
                            if end_frame is not None and 0 <= int(end_frame) < total_frames:
                                res_frame = int(end_frame)
                            else:
                                res_frame = max(0, min(total_frames - 1, int(round(clamped_end_ts * fps))))
                            if res_frame != cand_frame:
                                candidates.append(
                                    CandidateFrame(
                                        frame_idx=res_frame,
                                        timestamp_s=round(clamped_end_ts, 3),
                                        priority=2,
                                        importance_score=duration * 0.8,
                                        selection_reason=f"stopped_vehicle_resolution_track_{track_id}",
                                        associated_event_id=event_id,
                                        associated_track_id=track_id,
                                    )
                                )
                    elif start_frame is not None and 0 <= int(start_frame) < total_frames:
                        cand_frame = int(start_frame)
                        cand_ts = round(cand_frame / fps, 3)
                        candidates.append(
                            CandidateFrame(
                                frame_idx=cand_frame,
                                timestamp_s=cand_ts,
                                priority=1,
                                importance_score=duration,
                                selection_reason=f"stopped_vehicle_start_track_{track_id}",
                                associated_event_id=event_id,
                                associated_track_id=track_id,
                            )
                        )
                    else:
                        diagnostics.warnings.append(
                            f"Stopped vehicle event for track {track_id} has missing or non-finite timestamp and frame index; skipped."
                        )

                # Case 2: FPS is unavailable (fps is None)
                else:
                    if start_frame is not None and 0 <= int(start_frame) < total_frames:
                        cand_frame = int(start_frame)
                        event_id = f"stop_{track_id or idx}_f{cand_frame}"
                        candidates.append(
                            CandidateFrame(
                                frame_idx=cand_frame,
                                timestamp_s=None,
                                priority=1,
                                importance_score=duration,
                                selection_reason=f"stopped_vehicle_start_track_{track_id}",
                                associated_event_id=event_id,
                                associated_track_id=track_id,
                            )
                        )
                        if end_frame is not None and 0 <= int(end_frame) < total_frames and int(end_frame) != cand_frame:
                            candidates.append(
                                CandidateFrame(
                                    frame_idx=int(end_frame),
                                    timestamp_s=None,
                                    priority=2,
                                    importance_score=duration * 0.8,
                                    selection_reason=f"stopped_vehicle_resolution_track_{track_id}",
                                    associated_event_id=event_id,
                                    associated_track_id=track_id,
                                )
                            )
                        if config.include_pre_event_context:
                            diagnostics.warnings.append(
                                f"Pre-event context frame for track {track_id} skipped: video timing cannot be established reliably without FPS."
                            )
                    else:
                        diagnostics.warnings.append(
                            f"Cannot map timestamped stopped-vehicle event (track={track_id}, "
                            f"start_timestamp_s={start_ts}) to frame index: stream and metadata FPS are unavailable. "
                            "Assumed FPS mapping is prohibited."
                        )

        # B. Congestion & Occupancy Transitions (Priority = 2)
        if congestion_states:
            for cs in congestion_states:
                ts = getattr(cs, "timestamp_s", None)
                if ts is None and isinstance(cs, dict):
                    ts = cs.get("timestamp_s")
                cs_frame = getattr(cs, "frame_idx", None)
                if cs_frame is None and isinstance(cs, dict):
                    cs_frame = cs.get("frame_idx")
                level = getattr(cs, "level", None)
                if level is None and isinstance(cs, dict):
                    level = cs.get("level")
                occ = getattr(cs, "occupancy_ratio", 0.0)
                if occ is None and isinstance(cs, dict):
                    occ = cs.get("occupancy_ratio", 0.0)
                roi_id = getattr(cs, "roi_id", "roi")
                if roi_id is None and isinstance(cs, dict):
                    roi_id = cs.get("roi_id", "roi")

                level_str = str(level).lower()
                if "high" in level_str or "moderate" in level_str:
                    if fps is not None and fps > 0:
                        if ts is not None and math.isfinite(ts):
                            clamped_ts = max(0.0, min(duration_s, float(ts))) if duration_s is not None else max(0.0, float(ts))
                            frame_idx = max(0, min(total_frames - 1, int(round(clamped_ts * fps))))
                            candidates.append(
                                CandidateFrame(
                                    frame_idx=frame_idx,
                                    timestamp_s=round(clamped_ts, 3),
                                    priority=2,
                                    importance_score=float(occ or 0.0) * 10.0,
                                    selection_reason=f"congestion_{level_str}_{roi_id}",
                                    associated_event_id=f"congestion_{roi_id}_{clamped_ts:.2f}",
                                )
                            )
                        elif cs_frame is not None and 0 <= int(cs_frame) < total_frames:
                            frame_idx = int(cs_frame)
                            candidates.append(
                                CandidateFrame(
                                    frame_idx=frame_idx,
                                    timestamp_s=round(frame_idx / fps, 3),
                                    priority=2,
                                    importance_score=float(occ or 0.0) * 10.0,
                                    selection_reason=f"congestion_{level_str}_{roi_id}",
                                    associated_event_id=f"congestion_{roi_id}_f{frame_idx}",
                                )
                            )
                    else:
                        if cs_frame is not None and 0 <= int(cs_frame) < total_frames:
                            frame_idx = int(cs_frame)
                            candidates.append(
                                CandidateFrame(
                                    frame_idx=frame_idx,
                                    timestamp_s=None,
                                    priority=2,
                                    importance_score=float(occ or 0.0) * 10.0,
                                    selection_reason=f"congestion_{level_str}_{roi_id}",
                                    associated_event_id=f"congestion_{roi_id}_f{frame_idx}",
                                )
                            )
                        else:
                            diagnostics.warnings.append(
                                f"Cannot map congestion state (roi={roi_id}, timestamp_s={ts}) to frame index: "
                                "stream and metadata FPS are unavailable. Assumed FPS mapping is prohibited."
                            )

        # C. Periodic Scene Anchors (Priority = 3)
        if fps is not None and duration_s is not None and duration_s > 0:
            anchor_fractions = [0.05, 0.50, 0.95]
            if duration_s >= 15.0 and config.frame_budget >= 5:
                anchor_fractions = [0.05, 0.25, 0.50, 0.75, 0.95]

            for frac in anchor_fractions:
                anchor_frame = max(0, min(total_frames - 1, int(round((total_frames - 1) * frac))))
                anchor_ts = round(anchor_frame / fps, 3)
                candidates.append(
                    CandidateFrame(
                        frame_idx=anchor_frame,
                        timestamp_s=anchor_ts,
                        priority=3,
                        importance_score=1.0,
                        selection_reason=f"scene_anchor_pct_{int(frac * 100)}",
                    )
                )
        elif total_frames > 0:
            # Deterministic anchor sampling across frame indices when FPS is unavailable
            anchor_fractions = [0.05, 0.50, 0.95]
            if total_frames >= 300 and config.frame_budget >= 5:
                anchor_fractions = [0.05, 0.25, 0.50, 0.75, 0.95]

            for frac in anchor_fractions:
                anchor_frame = max(0, min(total_frames - 1, int(round((total_frames - 1) * frac))))
                candidates.append(
                    CandidateFrame(
                        frame_idx=anchor_frame,
                        timestamp_s=None,
                        priority=3,
                        importance_score=1.0,
                        selection_reason=f"scene_anchor_frame_{anchor_frame}",
                    )
                )

        return candidates

    def _suppress_duplicates(
        self,
        candidates: List[CandidateFrame],
        config: ExtractionConfigSchema,
        diagnostics: ExtractionDiagnosticsSchema,
    ) -> List[CandidateFrame]:
        """
        Suppresses redundant or near-duplicate candidate frames.
        
        Candidates separated by less than min_event_interval_s are pruned unless
        they represent distinct physical events. When timestamps are null, candidates
        at the same frame index are pruned unless they represent distinct events.
        """
        if not candidates:
            return []

        # Sort candidates deterministically:
        # 1. priority ascending (1 = highest priority)
        # 2. importance_score descending
        # 3. frame_idx ascending (deterministic tie-break without fabricating temporal ordering)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.priority,
                -c.importance_score,
                c.frame_idx,
                c.timestamp_s if c.timestamp_s is not None else -1.0,
            ),
        )

        accepted: List[CandidateFrame] = []
        for cand in sorted_candidates:
            collision = False
            for acc in accepted:
                # Both candidates have valid timestamps
                if cand.timestamp_s is not None and acc.timestamp_s is not None:
                    diff = abs(cand.timestamp_s - acc.timestamp_s)
                    if diff < config.min_event_interval_s:
                        cand_has_event = cand.associated_event_id is not None
                        acc_has_event = acc.associated_event_id is not None
                        if cand_has_event and acc_has_event:
                            if cand.associated_event_id == acc.associated_event_id:
                                collision = True
                                break
                            # Distinct events occurring simultaneously can coexist
                        else:
                            collision = True
                            break
                # Either candidate has a null timestamp: evaluate by exact frame index
                elif cand.frame_idx == acc.frame_idx:
                    cand_has_event = cand.associated_event_id is not None
                    acc_has_event = acc.associated_event_id is not None
                    if cand_has_event and acc_has_event:
                        if cand.associated_event_id == acc.associated_event_id:
                            collision = True
                            break
                        # Distinct events occurring at same frame index can coexist
                    else:
                        collision = True
                        break

            if collision:
                diagnostics.candidates_pruned_duplicate += 1
            else:
                accepted.append(cand)

        return accepted

    def _enforce_budget(
        self,
        candidates: List[CandidateFrame],
        config: ExtractionConfigSchema,
        diagnostics: ExtractionDiagnosticsSchema,
    ) -> List[CandidateFrame]:
        """
        Enforces maximum frame budget (K) using deterministic priority and temporal diversity.
        """
        if len(candidates) <= config.frame_budget:
            # If candidates are fewer than K, return all candidates sorted chronologically
            return sorted(candidates, key=lambda c: c.frame_idx)

        # Rank by:
        # 1. priority ascending (1 = highest priority)
        # 2. importance_score descending
        # 3. frame_idx ascending
        ranked = sorted(
            candidates,
            key=lambda c: (c.priority, -c.importance_score, c.frame_idx),
        )

        # Select top K
        selected = ranked[:config.frame_budget]
        diagnostics.candidates_pruned_budget += (len(candidates) - config.frame_budget)

        # Return sorted chronologically by frame_idx for sequential visual coherence
        return sorted(selected, key=lambda c: c.frame_idx)

    def _decode_and_encode_frames(
        self,
        cap: cv2.VideoCapture,
        candidates: List[CandidateFrame],
        fps: Optional[float],
        total_frames: int,
        config: ExtractionConfigSchema,
        diagnostics: ExtractionDiagnosticsSchema,
    ) -> List[ExtractedFrame]:
        """Seeks to candidate frame indices, decodes, resizes, and encodes to JPEG."""
        extracted: List[ExtractedFrame] = []

        for cand in candidates:
            target_idx = cand.frame_idx
            if target_idx < 0 or target_idx >= total_frames:
                target_idx = max(0, min(total_frames - 1, target_idx))
                diagnostics.warnings.append(
                    f"Candidate frame index {cand.frame_idx} out of range; clamped to {target_idx}"
                )

            # Seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                diagnostics.warnings.append(
                    f"OpenCV failed to decode frame at index {target_idx} (reason: {cand.selection_reason})"
                )
                continue

            # Query post-read position
            pos_after = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Evaluate decoder position semantics:
            # In OpenCV, CAP_PROP_POS_FRAMES after cap.read() returns the 0-based index of the NEXT frame to be read (n + 1).
            # Therefore, the decoded frame index is int(round(pos_after)) - 1.
            is_verified = False
            actual_idx = target_idx

            if pos_after is not None and math.isfinite(pos_after) and pos_after >= 1.0:
                reported_next_idx = int(round(pos_after))
                decoded_idx = reported_next_idx - 1

                # Check consistency against valid frame range
                if 0 <= decoded_idx < total_frames:
                    is_verified = True
                    actual_idx = decoded_idx
                    if actual_idx != target_idx:
                        diagnostics.warnings.append(
                            f"Seek position divergence on compressed container: requested frame {target_idx}, "
                            f"but decoder reported frame {actual_idx}. Recording actual decoded frame index {actual_idx}."
                        )
                else:
                    diagnostics.warnings.append(
                        f"Decoder reported out-of-range post-read position {pos_after} (derived frame index {decoded_idx}) "
                        f"for requested frame {target_idx} (total frames: {total_frames}). Decoder position is unverified."
                    )
            elif pos_after is not None and math.isfinite(pos_after) and pos_after == 0.0:
                diagnostics.warnings.append(
                    f"Decoder position did not advance after read (CAP_PROP_POS_FRAMES returned 0.0) for requested frame {target_idx}. "
                    "Frame correspondence cannot be verified via decoder query; retaining requested index as unverified."
                )
            else:
                diagnostics.warnings.append(
                    f"Decoder position query unavailable or invalid (CAP_PROP_POS_FRAMES returned {pos_after}) for requested frame {target_idx}. "
                    "Frame correspondence cannot be verified; retaining requested index as unverified."
                )

            if is_verified:
                diagnostics.verified_frame_count += 1
            else:
                diagnostics.unverified_frame_count += 1

            orig_h, orig_w = frame.shape[:2]

            # Aspect-ratio preserving downscaling if exceeding max_dimension
            if max(orig_h, orig_w) > config.max_dimension:
                scale = config.max_dimension / float(max(orig_h, orig_w))
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized_frame = frame
                new_w = orig_w
                new_h = orig_h

            # JPEG Encoding
            success, buffer = cv2.imencode(
                ".jpg",
                resized_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality],
            )
            if not success or buffer is None:
                diagnostics.warnings.append(f"Failed to encode frame {actual_idx} to JPEG")
                continue

            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")

            event_ids = [cand.associated_event_id] if cand.associated_event_id else []

            # If seek position diverged and FPS is reliably known, recompute timestamp;
            # if cand.timestamp_s is None or FPS is None, timestamp remains None.
            frame_ts = cand.timestamp_s
            if actual_idx != target_idx and fps is not None and fps > 0 and frame_ts is not None:
                frame_ts = round(actual_idx / fps, 3)

            extracted.append(
                ExtractedFrame(
                    frame_idx=actual_idx,
                    requested_frame_idx=target_idx,
                    is_verified=is_verified,
                    timestamp_s=frame_ts,
                    selection_reason=cand.selection_reason,
                    associated_event_ids=event_ids,
                    image_bytes=img_bytes,
                    image_base64=img_b64,
                    width=new_w,
                    height=new_h,
                )
            )

        return extracted
