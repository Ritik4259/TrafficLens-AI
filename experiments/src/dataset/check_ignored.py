import zipfile
import io
import pandas as pd
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    csv_bytes = zf.read("detrac_1_MVI_20011/annotations.csv")
    df = pd.read_csv(io.BytesIO(csv_bytes))
    ign_df = df[df['class'] == 'ignored']
    print("Ignored rows in MVI_20011:", len(ign_df))
    print("Ignored rows per frame count:", ign_df['filename'].value_counts().unique())
    # Compare ignored boxes on frame 1 vs frame 2
    f1 = ign_df[ign_df['filename'] == 'image000001.jpg'][['xmin', 'ymin', 'xmax', 'ymax']].values
    f2 = ign_df[ign_df['filename'] == 'image000002.jpg'][['xmin', 'ymin', 'xmax', 'ymax']].values
    print("f1 ignored boxes count:", len(f1))
    print("f2 ignored boxes count:", len(f2))
    print("Are ignored boxes identical between frame 1 and 2?", (f1 == f2).all())
