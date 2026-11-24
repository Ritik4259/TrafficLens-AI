/**
 * Common geometric and spatial types matching backend/app/schemas/common.py
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface BBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface CountingLineConfig {
  line_id: string;
  name: string;
  p1: [number, number];
  p2: [number, number];
  positive_direction_label?: string;
  negative_direction_label?: string;
  cooldown_frames?: number;
  min_track_age?: number;
}

export interface ROIConfig {
  roi_id: string;
  name: string;
  polygon: Array<[number, number]>;
  description?: string;
}
