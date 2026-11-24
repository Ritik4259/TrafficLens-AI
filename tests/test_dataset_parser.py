"""
Unit tests for UA-DETRAC dataset parser, inspector, and curator.
"""

import xml.etree.ElementTree as ET
import pytest
import numpy as np
from pathlib import Path

from experiments.src.dataset.parser import (
    BBox,
    VehicleAttribute,
    Target,
    FrameAnnotation,
    IgnoredRegion,
    SequenceAnnotation,
    parse_detrac_xml,
    CANONICAL_VEHICLE_TYPES,
)
from experiments.src.dataset.inspector import DatasetInspector
from experiments.src.dataset.curator import SequenceCurator
from experiments.src.dataset.extract_curated_dataset import validate_frame_contiguity


SAMPLE_XML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<sequence name="MVI_20011">
    <sequence_attribute camera_state="unstable" sence_condition="night" fog_status="clear"/>
    <ignored_region>
        <box left="200.0" top="100.0" width="300.0" height="150.0"/>
    </ignored_region>
    <frame num="1" density="2">
        <target_list>
            <target id="1">
                <box left="100.0" top="250.0" width="80.0" height="60.0"/>
                <attribute vehicle_type="car" color="black" orientation="180.0" truncation_ratio="0.0" speed="25.5" occluded_ratio="0.1"/>
            </target>
            <target id="2">
                <box left="300.0" top="300.0" width="120.0" height="90.0"/>
                <attribute vehicle_type="bus" color="blue" orientation="90.0" truncation_ratio="0.2" speed="18.0" occluded_ratio="0.0"/>
            </target>
        </target_list>
    </frame>
    <frame num="2" density="3">
        <target_list>
            <target id="1">
                <box left="105.0" top="255.0" width="82.0" height="61.0"/>
                <attribute vehicle_type="car" color="black" orientation="180.0" truncation_ratio="0.0" speed="25.0" occluded_ratio="0.1"/>
            </target>
            <target id="2">
                <box left="310.0" top="305.0" width="120.0" height="90.0"/>
                <attribute vehicle_type="bus" color="blue" orientation="90.0" truncation_ratio="0.2" speed="17.5" occluded_ratio="0.0"/>
            </target>
            <target id="3">
                <box left="500.0" top="400.0" width="95.0" height="75.0"/>
                <attribute vehicle_type="van" color="white" orientation="45.0" truncation_ratio="0.0" speed="30.0" occluded_ratio="0.3"/>
            </target>
        </target_list>
    </frame>
</sequence>
"""


@pytest.fixture
def sample_xml_file(tmp_path: Path) -> Path:
    xml_path = tmp_path / "MVI_20011.xml"
    xml_path.write_text(SAMPLE_XML_CONTENT, encoding="utf-8")
    return xml_path


class TestBBox:
    def test_bbox_properties(self):
        box = BBox(left=10.0, top=20.0, width=30.0, height=40.0)
        assert box.x1 == 10.0
        assert box.y1 == 20.0
        assert box.x2 == 40.0
        assert box.y2 == 60.0
        assert box.area == 1200.0
        assert box.center == (25.0, 40.0)
        assert box.to_xyxy() == (10.0, 20.0, 40.0, 60.0)
        assert box.to_xywh() == (10.0, 20.0, 30.0, 40.0)

    def test_bbox_clipping(self):
        box = BBox(left=-10.0, top=-5.0, width=1010.0, height=605.0)
        clipped = box.clip(max_width=960.0, max_height=540.0)
        assert clipped.x1 == 0.0
        assert clipped.y1 == 0.0
        assert clipped.x2 == 960.0
        assert clipped.y2 == 540.0
        assert clipped.width == 960.0
        assert clipped.height == 540.0

    def test_bbox_iou_identical(self):
        b1 = BBox(0, 0, 10, 10)
        b2 = BBox(0, 0, 10, 10)
        assert pytest.approx(b1.iou(b2)) == 1.0

    def test_bbox_iou_disjoint(self):
        b1 = BBox(0, 0, 10, 10)
        b2 = BBox(20, 20, 10, 10)
        assert b1.iou(b2) == 0.0

    def test_bbox_iou_partial_overlap(self):
        b1 = BBox(0, 0, 10, 10)
        b2 = BBox(5, 0, 10, 10)
        assert pytest.approx(b1.iou(b2)) == 50.0 / 150.0


class TestIgnoredRegionBoundaryCases:
    """Boundary-case testing for ignore-region matching rules."""

    def test_fully_contained_box_is_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=100.0, top=100.0, width=200.0, height=200.0)])
        box = BBox(left=120.0, top=120.0, width=50.0, height=50.0)
        assert ign.get_overlap_ratio(box) == 1.0
        assert ign.is_box_ignored(box, threshold=0.5) is True

    def test_completely_disjoint_box_is_not_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=100.0, top=100.0, width=200.0, height=200.0)])
        box = BBox(left=400.0, top=400.0, width=50.0, height=50.0)
        assert ign.get_overlap_ratio(box) == 0.0
        assert ign.is_box_ignored(box, threshold=0.5) is False

    def test_tangent_edge_boundary_is_not_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=100.0, top=100.0, width=200.0, height=200.0)])
        box = BBox(left=300.0, top=100.0, width=50.0, height=50.0)
        assert ign.get_overlap_ratio(box) == 0.0
        assert ign.is_box_ignored(box, threshold=0.5) is False

    def test_below_threshold_49_percent_is_not_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=0.0, top=0.0, width=100.0, height=100.0)])
        box = BBox(left=51.0, top=0.0, width=100.0, height=100.0)
        assert pytest.approx(ign.get_overlap_ratio(box)) == 0.49
        assert ign.is_box_ignored(box, threshold=0.5) is False

    def test_above_threshold_51_percent_is_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=0.0, top=0.0, width=100.0, height=100.0)])
        box = BBox(left=49.0, top=0.0, width=100.0, height=100.0)
        assert pytest.approx(ign.get_overlap_ratio(box)) == 0.51
        assert ign.is_box_ignored(box, threshold=0.5) is True

    def test_adjacent_multi_box_ignore_union(self):
        ign1 = BBox(left=0.0, top=0.0, width=50.0, height=100.0)
        ign2 = BBox(left=50.0, top=0.0, width=50.0, height=100.0)
        ign = IgnoredRegion(boxes=[ign1, ign2])

        box = BBox(left=20.0, top=0.0, width=60.0, height=100.0)
        assert pytest.approx(ign.get_overlap_ratio(box)) == 1.0
        assert ign.is_box_ignored(box, threshold=0.5) is True

    def test_center_inside_but_minor_overlap_not_falsely_ignored(self):
        ign = IgnoredRegion(boxes=[BBox(left=0.0, top=0.0, width=100.0, height=100.0)])
        box = BBox(left=80.0, top=0.0, width=100.0, height=100.0)
        assert pytest.approx(ign.get_overlap_ratio(box)) == 0.20
        assert ign.is_box_ignored(box, threshold=0.5) is False

    def test_static_vs_frame_replicated_ignore_counts(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        # Sequence has 1 static ignore box, 2 frames
        assert seq.static_ignored_box_count == 1
        assert seq.total_frame_replicated_ignored_boxes == 2
        assert len(seq.get_frame_ignored_boxes(1)) == 1


class TestXMLParser:
    def test_parse_valid_xml(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        assert seq.name == "MVI_20011"
        assert seq.camera_state == "unstable"
        assert seq.scene_condition == "night"
        assert seq.fog_status == "clear"
        assert len(seq.ignored_region.boxes) == 1
        assert seq.ignored_region.boxes[0].to_xywh() == (200.0, 100.0, 300.0, 150.0)

        assert seq.total_frames == 2
        assert seq.total_boxes == 5
        assert seq.get_all_target_ids() == {1, 2, 3}

    def test_category_preservation_no_silent_merges(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        frame2 = seq.frames[2]
        classes = {t.attribute.vehicle_type for t in frame2.targets}
        assert "car" in classes
        assert "bus" in classes
        assert "van" in classes
        assert classes.issubset(CANONICAL_VEHICLE_TYPES)
        # Verify 'vehicle' is also recognized as valid canonical category
        assert "vehicle" in CANONICAL_VEHICLE_TYPES

    def test_frame_image_naming(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        assert seq.get_frame_image_name(1) == "image000001.jpg"
        assert seq.get_frame_image_name(25) == "image000025.jpg"
        assert seq.get_frame_image_name(1, prefix="img", digits=5) == "img00001.jpg"
        assert seq.get_frame_image_name(1340, prefix="img", digits=5) == "img01340.jpg"

        img_path = seq.get_frame_image_path("data/raw/images/MVI_20011", 1)
        assert img_path.name == "image000001.jpg"

    def test_trajectory_extraction(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        traj1 = seq.get_trajectory(target_id=1)
        assert len(traj1) == 2
        assert traj1[0][0] == 1
        assert traj1[1][0] == 2
        assert traj1[0][1].left == 100.0
        assert traj1[1][1].left == 105.0

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            parse_detrac_xml("non_existent_file.xml")

    def test_malformed_xml_raises_error(self, tmp_path: Path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<sequence><unclosed>", encoding="utf-8")
        with pytest.raises(ET.ParseError):
            parse_detrac_xml(bad_xml)


class TestContiguityValidation:
    def test_valid_contiguous_frames(self):
        validate_frame_contiguity([1, 2, 3, 4, 5], "MVI_TEST")

    def test_missing_frames_raises_error(self):
        with pytest.raises(ValueError, match="missing frames"):
            validate_frame_contiguity([1, 2, 4, 5], "MVI_TEST")

    def test_duplicate_frames_raises_error(self):
        with pytest.raises(ValueError, match="duplicate frame"):
            validate_frame_contiguity([1, 2, 2, 3], "MVI_TEST")


class TestInspectorAndCurator:
    def test_inspector_statistics(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        stats = DatasetInspector.inspect_sequence(seq)
        assert stats.name == "MVI_20011"
        assert stats.unique_vehicles == 3
        assert stats.total_bounding_boxes == 5
        assert stats.class_distribution == {"car": 2, "bus": 2, "van": 1}
        assert stats.density_min == 2
        assert stats.density_max == 3
        assert stats.ignored_box_count == 1

    def test_render_annotations(self, sample_xml_file: Path):
        seq = parse_detrac_xml(sample_xml_file)
        frame1 = seq.frames[1]
        dummy_img = np.zeros((540, 960, 3), dtype=np.uint8)
        vis = DatasetInspector.render_annotations(
            dummy_img, frame1, seq.ignored_region, draw_ignore=True, draw_labels=True
        )
        assert vis.shape == (540, 960, 3)
        assert np.count_nonzero(vis) > 0

    def test_curator_default_manifest_uniqueness_and_provisional_status(self):
        curator = SequenceCurator(seed=42)
        manifest = curator.get_default_curated_subset()
        assert manifest.total_sequences == 8
        assert manifest.status == "provisional"

        all_seqs = manifest.splits["train"] + manifest.splits["val"] + manifest.splits["test"]
        assert len(all_seqs) == len(set(all_seqs)) == 8

        assert len(manifest.split_conditions["train"]) == 4
        assert len(manifest.split_conditions["val"]) == 2
        assert len(manifest.split_conditions["test"]) == 2

    def test_curator_reproducibility(self):
        c1 = SequenceCurator(seed=123)
        c2 = SequenceCurator(seed=123)
        m1 = c1.get_default_curated_subset()
        m2 = c2.get_default_curated_subset()
        assert m1.splits == m2.splits
