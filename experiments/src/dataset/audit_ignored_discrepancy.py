import pandas as pd
from pathlib import Path

annotations_dir = Path("data/raw/annotations")
sequences = [
    "MVI_39801",
    "MVI_40141",
    "MVI_63521",
    "MVI_63561",
    "MVI_20012",
    "MVI_39811",
    "MVI_20011",
    "MVI_40131",
]

print("=" * 80)
print("AUDIT OF IGNORED ANNOTATIONS ACROSS ALL 8 SEQUENCES")
print("=" * 80)

total_ignored_rows = 0
total_unique_boxes = 0

for seq in sequences:
    csv_path = annotations_dir / f"{seq}_annotations.csv"
    df = pd.read_csv(csv_path)
    ign_df = df[df["class"] == "ignored"]
    
    total_ignored_rows += len(ign_df)
    
    # Check unique frames
    unique_frames = df["filename"].nunique()
    ign_frames = ign_df["filename"].nunique()
    
    # Counts of ignored boxes per frame
    per_frame_counts = ign_df["filename"].value_counts()
    min_ign_per_frame = per_frame_counts.min() if not per_frame_counts.empty else 0
    max_ign_per_frame = per_frame_counts.max() if not per_frame_counts.empty else 0
    
    # Unique coordinate sets across frames
    unique_boxes = ign_df[["xmin", "ymin", "xmax", "ymax"]].drop_duplicates()
    total_unique_boxes += len(unique_boxes)
    
    print(f"\nSequence: {seq}")
    print(f"  Total frames: {unique_frames}")
    print(f"  Frames with ignored annotations: {ign_frames}")
    print(f"  Ignored rows in CSV: {len(ign_df)}")
    print(f"  Ignored boxes per frame: min={min_ign_per_frame}, max={max_ign_per_frame}")
    print(f"  Unique spatial boxes: {len(unique_boxes)}")
    print(f"  Ratio (ignored_rows / frames): {len(ign_df) / unique_frames:.2f}")

print("\n" + "=" * 80)
print(f"Total Ignored Rows in CSVs: {total_ignored_rows}")
print(f"Total Unique Spatial Ignored Boxes across 8 sequences: {total_unique_boxes}")
