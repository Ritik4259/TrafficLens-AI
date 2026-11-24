import zipfile
import pickle
import io
from pathlib import Path

desktop = Path(r"c:\Users\bhard\Desktop")
train_zip = desktop / "ua_detrac_training_set.zip"

with zipfile.ZipFile(train_zip, 'r') as zf:
    pkl_bytes = zf.read("detrac_25_MVI_40152/annotations/.frames_list.pkl")
    data = pickle.loads(pkl_bytes)
    print("PKL content:", data)
