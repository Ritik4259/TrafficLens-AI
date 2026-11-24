"""
UA-DETRAC XML & CSV Dataset Parser.

Parses official UA-DETRAC XML annotations and CSV exports into strongly typed Python objects,
preserving all original bounding box coordinates, trajectory IDs, vehicle attributes,
and ignored regions. Never silently merges vehicle categories.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


CANONICAL_VEHICLE_TYPES: Set[str] = {"car", "bus", "van", "others", "vehicle"}
DEFAULT_FRAME_WIDTH: float = 960.0
DEFAULT_FRAME_HEIGHT: float = 540.0


@dataclass(frozen=True)
class BBox:
    """Bounding box in absolute image coordinates (left, top, width, height)."""
    left: float
    top: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.left

    @property
    def y1(self) -> float:
        return self.top

    @property
    def x2(self) -> float:
        return self.left + self.width

    @property
    def y2(self) -> float:
        return self.top + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Returns (left, top, width, height)."""
        return (self.left, self.top, self.width, self.height)

    def clip(self, max_width: float = DEFAULT_FRAME_WIDTH, max_height: float = DEFAULT_FRAME_HEIGHT) -> BBox:
        """Clips bounding box to image boundaries [0, max_width] x [0, max_height]."""
        cx1 = max(0.0, min(self.x1, max_width))
        cy1 = max(0.0, min(self.y1, max_height))
        cx2 = max(0.0, min(self.x2, max_width))
        cy2 = max(0.0, min(self.y2, max_height))
        return BBox(left=cx1, top=cy1, width=max(0.0, cx2 - cx1), height=max(0.0, cy2 - cy1))

    def iou(self, other: BBox) -> float:
        """Computes Intersection over Union (IoU) with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def intersection_with(self, other: BBox) -> Optional[BBox]:
        """Returns the intersecting BBox with another box, or None if disjoint."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        if iw <= 0 or ih <= 0:
            return None
        return BBox(left=ix1, top=iy1, width=iw, height=ih)

    def intersection_area_with(self, other: BBox) -> float:
        """Computes the raw intersection area with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return iw * ih

    def intersection_ratio_with(self, other: BBox) -> float:
        """Computes intersection area divided by self.area."""
        if self.area <= 0:
            return 0.0
        return self.intersection_area_with(other) / self.area


@dataclass
class VehicleAttribute:
    """Attributes associated with an annotated vehicle in a specific frame."""
    vehicle_type: str
    color: Optional[str] = None
    orientation: Optional[float] = None
    truncation_ratio: Optional[float] = None
    speed: Optional[float] = None
    occluded_ratio: Optional[float] = None

    def __post_init__(self):
        # Normalize casing but do NOT merge categories
        self.vehicle_type = self.vehicle_type.strip().lower()


@dataclass
class Target:
    """A vehicle detection associated with a trajectory ID in a frame."""
    target_id: int
    bbox: BBox
    attribute: VehicleAttribute


@dataclass
class FrameAnnotation:
    """Annotations for a single video frame."""
    frame_num: int
    density: int
    targets: List[Target] = field(default_factory=list)

    @property
    def target_count(self) -> int:
        return len(self.targets)


@dataclass
class IgnoredRegion:
    """
    Ignored regions for a sequence where detections are not penalized.
    
    Evaluation Semantics:
    In the official UA-DETRAC benchmark, detections falling predominantly inside
    an ignored region are masked out of evaluation (neither rewarded as TP nor penalized as FP).
    """
    boxes: List[BBox] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of unique static spatial ignore boxes in this region."""
        return len(self.boxes)

    def contains_point(self, x: float, y: float) -> bool:
        """Checks if a point (x, y) lies inside any ignored box."""
        for box in self.boxes:
            if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
                return True
        return False

    def get_overlap_ratio(self, bbox: BBox) -> float:
        """
        Calculates the ratio of bbox area covered by the union of ignored boxes:
        overlap_ratio = Area(bbox ∩ ⋃ Ignored_i) / Area(bbox).
        """
        if bbox.area <= 0 or not self.boxes:
            return 0.0

        intersecting = []
        for ign in self.boxes:
            inter = bbox.intersection_with(ign)
            if inter is not None and inter.area > 0:
                intersecting.append(inter)

        if not intersecting:
            return 0.0

        if len(intersecting) == 1:
            return intersecting[0].area / bbox.area

        # Compute exact union area using 1D x-strip interval decomposition
        x_coords = sorted(set([b.x1 for b in intersecting] + [b.x2 for b in intersecting]))
        total_inter_area = 0.0
        for i in range(len(x_coords) - 1):
            x_left = x_coords[i]
            x_right = x_coords[i + 1]
            dx = x_right - x_left
            if dx <= 0:
                continue

            y_intervals = []
            for b in intersecting:
                if b.x1 <= x_left and b.x2 >= x_right:
                    y_intervals.append((b.y1, b.y2))

            if not y_intervals:
                continue

            y_intervals.sort(key=lambda item: item[0])
            merged_y = []
            cur_y1, cur_y2 = y_intervals[0]
            for next_y1, next_y2 in y_intervals[1:]:
                if next_y1 <= cur_y2:
                    cur_y2 = max(cur_y2, next_y2)
                else:
                    merged_y.append((cur_y1, cur_y2))
                    cur_y1, cur_y2 = next_y1, next_y2
            merged_y.append((cur_y1, cur_y2))

            dy_total = sum(y2 - y1 for y1, y2 in merged_y)
            total_inter_area += dx * dy_total

        return min(1.0, total_inter_area / bbox.area)

    def is_box_ignored(self, bbox: BBox, threshold: float = 0.5) -> bool:
        """
        Determines if a bounding box should be treated as ignored.
        Returns True if the overlap ratio Area(bbox ∩ Ignored) / Area(bbox) >= threshold.
        """
        return self.get_overlap_ratio(bbox) >= threshold


@dataclass
class SequenceAnnotation:
    """Full annotations for a UA-DETRAC video sequence."""
    name: str
    camera_state: str
    scene_condition: str
    fog_status: str
    ignored_region: IgnoredRegion
    frames: Dict[int, FrameAnnotation] = field(default_factory=dict)

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def total_boxes(self) -> int:
        return sum(f.target_count for f in self.frames.values())

    @property
    def static_ignored_box_count(self) -> int:
        """Returns the number of unique spatial ignored bounding boxes defined for this sequence."""
        return self.ignored_region.count

    @property
    def total_frame_replicated_ignored_boxes(self) -> int:
        """
        Returns the total number of frame-replicated ignored boxes (frames * static_ignored_box_count),
        matching the count in per-frame Pascal VOC annotations and raw CSV exports.
        """
        return self.total_frames * self.static_ignored_box_count

    def get_frame_ignored_boxes(self, frame_num: int) -> List[BBox]:
        """Returns the list of ignored boxes active for a specific frame."""
        return list(self.ignored_region.boxes)

    def get_all_target_ids(self) -> Set[int]:
        """Returns set of all unique vehicle trajectory IDs in this sequence."""
        ids: Set[int] = set()
        for f in self.frames.values():
            for t in f.targets:
                ids.add(t.target_id)
        return ids

    def get_trajectory(self, target_id: int) -> List[Tuple[int, BBox, VehicleAttribute]]:
        """Extracts the full trajectory for a specific vehicle ID as (frame_num, bbox, attr)."""
        trajectory = []
        for frame_num in sorted(self.frames.keys()):
            for target in self.frames[frame_num].targets:
                if target.target_id == target_id:
                    trajectory.append((frame_num, target.bbox, target.attribute))
                    break
        return trajectory

    @staticmethod
    def get_frame_image_name(frame_num: int, prefix: str = "image", digits: int = 6) -> str:
        """
        Returns a frame filename. Default: image%06d.jpg (Kaggle archive format),
        also supports img%05d.jpg (official DETRAC portal format).
        """
        return f"{prefix}{frame_num:0{digits}d}.jpg"

    def get_frame_image_path(self, sequence_images_dir: Path | str, frame_num: int) -> Path:
        """
        Returns the path to a frame's JPEG image, dynamically detecting filename format.
        """
        seq_dir = Path(sequence_images_dir)
        for candidate in [
            f"image{frame_num:06d}.jpg",
            f"img{frame_num:05d}.jpg",
            f"image{frame_num:05d}.jpg",
            f"img{frame_num:06d}.jpg",
        ]:
            p = seq_dir / candidate
            if p.exists():
                return p
        return seq_dir / f"image{frame_num:06d}.jpg"


def _parse_float(val: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_box_element(elem: ET.Element) -> BBox:
    left = float(elem.attrib["left"])
    top = float(elem.attrib["top"])
    width = float(elem.attrib["width"])
    height = float(elem.attrib["height"])
    return BBox(left=left, top=top, width=width, height=height)


def parse_detrac_xml(file_path: Path | str) -> SequenceAnnotation:
    """
    Parses a single UA-DETRAC XML annotation file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag != "sequence":
        raise ValueError(f"Root tag must be <sequence>, got <{root.tag}> in {path.name}")

    sequence_name = root.attrib.get("name", path.stem)

    seq_attr_elem = root.find("sequence_attribute")
    if seq_attr_elem is not None:
        camera_state = seq_attr_elem.attrib.get("camera_state", "unknown")
        scene_condition = seq_attr_elem.attrib.get(
            "sence_condition", seq_attr_elem.attrib.get("scene_condition", "unknown")
        )
        fog_status = seq_attr_elem.attrib.get("fog_status", "unknown")
    else:
        camera_state, scene_condition, fog_status = "unknown", "unknown", "unknown"

    ignored_boxes: List[BBox] = []
    ignored_elem = root.find("ignored_region")
    if ignored_elem is not None:
        for box_elem in ignored_elem.findall("box"):
            try:
                ignored_boxes.append(_parse_box_element(box_elem))
            except (KeyError, ValueError):
                continue
    ignored_region = IgnoredRegion(boxes=ignored_boxes)

    frames_dict: Dict[int, FrameAnnotation] = {}
    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.attrib["num"])
        density = int(frame_elem.attrib.get("density", "0"))

        targets: List[Target] = []
        target_list_elem = frame_elem.find("target_list")
        if target_list_elem is not None:
            for target_elem in target_list_elem.findall("target"):
                target_id = int(target_elem.attrib["id"])
                box_elem = target_elem.find("box")
                if box_elem is None:
                    continue
                bbox = _parse_box_element(box_elem)

                attr_elem = target_elem.find("attribute")
                if attr_elem is not None:
                    vtype = attr_elem.attrib.get("vehicle_type", "vehicle")
                    color = attr_elem.attrib.get("color")
                    orientation = _parse_float(attr_elem.attrib.get("orientation"))
                    truncation = _parse_float(attr_elem.attrib.get("truncation_ratio"))
                    speed = _parse_float(attr_elem.attrib.get("speed"))
                    occlusion = _parse_float(attr_elem.attrib.get("occluded_ratio"))
                else:
                    vtype = "vehicle"
                    color, orientation, truncation, speed, occlusion = None, None, None, None, None

                attribute = VehicleAttribute(
                    vehicle_type=vtype,
                    color=color,
                    orientation=orientation,
                    truncation_ratio=truncation,
                    speed=speed,
                    occluded_ratio=occlusion,
                )
                targets.append(Target(target_id=target_id, bbox=bbox, attribute=attribute))

        frames_dict[frame_num] = FrameAnnotation(
            frame_num=frame_num,
            density=density,
            targets=targets,
        )

    return SequenceAnnotation(
        name=sequence_name,
        camera_state=camera_state,
        scene_condition=scene_condition,
        fog_status=fog_status,
        ignored_region=ignored_region,
        frames=frames_dict,
    )


def parse_all_sequences(annotations_dir: Path | str) -> Dict[str, SequenceAnnotation]:
    """
    Parses all XML annotation files within a directory.
    """
    ann_dir = Path(annotations_dir)
    if not ann_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {ann_dir}")

    sequences: Dict[str, SequenceAnnotation] = {}
    for xml_path in sorted(ann_dir.glob("*.xml")):
        seq = parse_detrac_xml(xml_path)
        sequences[seq.name] = seq

    return sequences
