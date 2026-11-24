import zipfile
import io
import pandas as pd
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    csv_bytes = zf.read("detrac_1_MVI_20011/annotations.csv")
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print("Columns in annotations.csv:", df.columns.tolist())
    print("Unique classes in detrac_1_MVI_20011/annotations.csv:", df['class'].value_counts())
    print("\nSample rows:")
    print(df.head(10))
    print(df[df['class'] != 'vehicle'].head(10))
