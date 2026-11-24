import zipfile
import io
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    # Read first 15 lines of annotations.csv for MVI_20011
    csv_bytes = zf.read("detrac_1_MVI_20011/annotations.csv")
    lines = csv_bytes.decode('utf-8', errors='ignore').splitlines()[:15]
    print("--- detrac_1_MVI_20011/annotations.csv header & sample lines ---")
    for l in lines:
        print(l)
    
    print("\n--- detrac_1_MVI_20011/annotations.zip inner contents ---")
    inner_zip_bytes = zf.read("detrac_1_MVI_20011/annotations.zip")
    with zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r') as inner_zf:
        for name in inner_zf.namelist()[:20]:
            print(f"  {name} (size: {inner_zf.getinfo(name).file_size})")
        if len(inner_zf.namelist()) > 20:
            print(f"  ... total inner files: {len(inner_zf.namelist())}")
