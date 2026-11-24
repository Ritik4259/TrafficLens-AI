"""
Geometric operations for traffic analytics.

Implements segment intersection, orientation, crossing direction determination,
point-in-polygon tests, Sutherland-Hodgman polygon clipping, and intersection area calculations.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import math

from experiments.src.dataset.parser import BBox


Point2D = Tuple[float, float]
Polygon2D = List[Point2D]


def orientation(p: Point2D, q: Point2D, r: Point2D) -> int:
    """
    Computes orientation of triplet (p, q, r).
    Returns:
         0 -> Collinear
         1 -> Clockwise
         2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """Checks if point q lies on segment pr."""
    return (
        q[0] <= max(p[0], r[0]) + 1e-9
        and q[0] >= min(p[0], r[0]) - 1e-9
        and q[1] <= max(p[1], r[1]) + 1e-9
        and q[1] >= min(p[1], r[1]) - 1e-9
    )


def segments_intersect(
    p1: Point2D,
    p2: Point2D,
    p3: Point2D,
    p4: Point2D,
) -> Tuple[bool, Optional[Point2D]]:
    """
    Determines whether line segment (p1, p2) intersects segment (p3, p4).
    
    Returns:
        (intersects: bool, intersection_point: Optional[Point2D])
    """
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # General intersection case
    if o1 != o2 and o3 != o4:
        # Solve parametric intersection
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p4[0] - p3[0]
        dy2 = p4[1] - p3[1]

        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) > 1e-9:
            t = ((p3[0] - p1[0]) * dy2 - (p3[1] - p1[1]) * dx2) / denom
            ix = p1[0] + t * dx1
            iy = p1[1] + t * dy1
            return True, (ix, iy)
        return True, p3

    # Collinear special cases
    if o1 == 0 and on_segment(p1, p3, p2):
        return True, p3
    if o2 == 0 and on_segment(p1, p4, p2):
        return True, p4
    if o3 == 0 and on_segment(p3, p1, p4):
        return True, p1
    if o4 == 0 and on_segment(p3, p2, p4):
        return True, p2

    return False, None


def determine_crossing_direction(
    prev_pos: Point2D,
    curr_pos: Point2D,
    line_p1: Point2D,
    line_p2: Point2D,
    positive_label: str = "downstream",
    negative_label: str = "upstream",
) -> str:
    """
    Determines crossing direction across a directed line segment (line_p1 -> line_p2).
    
    Uses the dot product of movement vector (curr_pos - prev_pos) with the left-normal
    vector of the line (-dy, dx). Positive dot product corresponds to crossing to the left
    of the directed line.
    """
    line_dx = line_p2[0] - line_p1[0]
    line_dy = line_p2[1] - line_p1[1]

    # Normal vector pointing to the left of the directed line
    normal_x = -line_dy
    normal_y = line_dx

    # Movement vector
    move_dx = curr_pos[0] - prev_pos[0]
    move_dy = curr_pos[1] - prev_pos[1]

    dot = move_dx * normal_x + move_dy * normal_y
    if dot >= 0:
        return positive_label
    return negative_label


def point_in_polygon(point: Point2D, polygon: Polygon2D) -> bool:
    """
    Determines if a 2D point lies inside or on the boundary of a polygon
    using the ray-casting algorithm (even-odd rule).
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def polygon_area(polygon: Polygon2D) -> float:
    """Calculates the area of a polygon using the Shoelace formula."""
    if len(polygon) < 3:
        return 0.0

    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0


def box_to_polygon(bbox: BBox) -> Polygon2D:
    """Converts a BBox into a 4-vertex polygon ordered clockwise."""
    return [
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y1),
        (bbox.x2, bbox.y2),
        (bbox.x1, bbox.y2),
    ]


def _clip_polygon_against_edge(
    poly: Polygon2D,
    edge_p1: Point2D,
    edge_p2: Point2D,
) -> Polygon2D:
    """Clips polygon against a directed line edge (keeps vertices to the left/inside)."""
    clipped: Polygon2D = []
    if not poly:
        return clipped

    def is_inside(p: Point2D) -> bool:
        # Cross product: (edge_p2 - edge_p1) x (p - edge_p1) >= 0 means left of edge
        return (edge_p2[0] - edge_p1[0]) * (p[1] - edge_p1[1]) - (edge_p2[1] - edge_p1[1]) * (p[0] - edge_p1[0]) >= -1e-9

    def line_intersection(p1: Point2D, p2: Point2D) -> Point2D:
        dc = (edge_p1[0] - edge_p2[0], edge_p1[1] - edge_p2[1])
        dp = (p1[0] - p2[0], p1[1] - p2[1])
        n1 = edge_p1[0] * edge_p2[1] - edge_p1[1] * edge_p2[0]
        n2 = p1[0] * p2[1] - p1[1] * p2[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-9:
            return p1
        ix = (n1 * dp[0] - n2 * dc[0]) / denom
        iy = (n1 * dp[1] - n2 * dc[1]) / denom
        return (ix, iy)

    s = poly[-1]
    for e in poly:
        if is_inside(e):
            if is_inside(s):
                clipped.append(e)
            else:
                clipped.append(line_intersection(s, e))
                clipped.append(e)
        elif is_inside(s):
            clipped.append(line_intersection(s, e))
        s = e

    return clipped


def sutherland_hodgman_polygon_clipping(
    subject_poly: Polygon2D,
    clip_poly: Polygon2D,
) -> Polygon2D:
    """
    Clips subject_poly against convex clip_poly using Sutherland-Hodgman algorithm.
    """
    output_poly = list(subject_poly)
    n = len(clip_poly)
    for i in range(n):
        if not output_poly:
            break
        edge_p1 = clip_poly[i]
        edge_p2 = clip_poly[(i + 1) % n]
        output_poly = _clip_polygon_against_edge(output_poly, edge_p1, edge_p2)

    return output_poly


def box_polygon_intersection_area(bbox: BBox, roi_poly: Polygon2D) -> float:
    """
    Calculates the intersection area between a bounding box and an ROI polygon.
    Clips the box against the ROI (if ROI is convex) or clips ROI against box.
    """
    if bbox.area <= 0 or len(roi_poly) < 3:
        return 0.0

    box_poly = box_to_polygon(bbox)
    # Clip ROI against the axis-aligned rectangular box (the box is guaranteed convex)
    clipped = sutherland_hodgman_polygon_clipping(roi_poly, box_poly)
    return polygon_area(clipped)
