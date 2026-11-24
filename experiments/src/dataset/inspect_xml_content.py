import zipfile
import io
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    inner_zip_bytes = zf.read("detrac_1_MVI_20011/annotations.zip")
    with zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r') as inner_zf:
        xml_content = inner_zf.read("image000001.xml").decode('utf-8', errors='ignore')
        print("--- detrac_1_MVI_20011/annotations.zip -> image000001.xml ---")
        print(xml_content[:1500])
