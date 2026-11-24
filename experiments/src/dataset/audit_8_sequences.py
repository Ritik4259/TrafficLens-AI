import zipfile
import io
import re
import pandas as pd
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

TARGET_SEQS = [
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
print("AUDIT OF 8 CURATED SEQUENCES IN ua_detrac_training_set.zip")
print("=" * 80)

seq_prefix_map = {}
with zipfile.ZipFile(train_zip, 'r') as zf:
    names = zf.namelist()
    for name in names:
        for t in TARGET_SEQS:
            if f"_{t}/" in name:
                prefix = name.split("/")[0]
                seq_prefix_map[t] = prefix

    print(f"Prefix mapping for all 8 sequences:")
    for t in TARGET_SEQS:
        print(f"  {t} -> {seq_prefix_map.get(t)}")

    results = []
    for t in TARGET_SEQS:
        prefix = seq_prefix_map[t]
        jpg_files = [n for n in names if n.startswith(prefix + "/") and n.lower().endswith(".jpg")]
        csv_name = f"{prefix}/annotations.csv"
        annot_zip_name = f"{prefix}/annotations.zip"
        
        has_csv = csv_name in names
        has_annot_zip = annot_zip_name in names
        
        num_xml = 0
        if has_annot_zip:
            inner_bytes = zf.read(annot_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes), 'r') as inner_zf:
                num_xml = len([x for x in inner_zf.namelist() if x.endswith(".xml")])
        
        num_vehicles = 0
        num_ignored = 0
        unique_ids = set()
        if has_csv:
            csv_bytes = zf.read(csv_name)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            counts = df['class'].value_counts().to_dict()
            num_vehicles = counts.get('vehicle', 0)
            num_ignored = counts.get('ignored', 0)
            vehicle_df = df[df['class'] == 'vehicle']
            unique_ids = set(vehicle_df['target_id'].dropna().astype(int).unique())

        results.append({
            "sequence": t,
            "prefix": prefix,
            "frames": len(jpg_files),
            "csv_rows": num_vehicles + num_ignored,
            "vehicle_boxes": num_vehicles,
            "ignored_boxes": num_ignored,
            "unique_targets": len(unique_ids),
            "xml_files": num_xml,
            "sample_img": jpg_files[0].split("/")[-1] if jpg_files else None,
            "sample_img_last": jpg_files[-1].split("/")[-1] if jpg_files else None,
        })

print("\nDetailed Summary Table:")
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
