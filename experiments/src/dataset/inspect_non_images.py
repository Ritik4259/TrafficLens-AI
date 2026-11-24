import zipfile
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    for name in zf.namelist():
        ext = Path(name).suffix.lower()
        if ext in [".csv", ".zip", ".pkl", ".bin"] or "xml" in name.lower() or "annot" in name.lower():
            info = zf.getinfo(name)
            print(f"{name} (size: {info.file_size} bytes)")
