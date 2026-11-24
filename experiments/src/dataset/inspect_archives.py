import zipfile
import re
from pathlib import Path
from collections import Counter, defaultdict

desktop = Path(r"c:\Users\bhard\Desktop")
zips = [
    desktop / "ua_detrac_training_set.zip",
    desktop / "ua_detrac_testing_set.zip",
]

TARGET_SEQUENCES = [
    "MVI_39801",
    "MVI_40141",
    "MVI_63521",
    "MVI_63561",
    "MVI_20012",
    "MVI_39811",
    "MVI_20011",
    "MVI_40131",
]

for zip_path in zips:
    print("=" * 80)
    print(f"Archive: {zip_path.name}")
    print(f"Size: {zip_path.stat().st_size / (1024**3):.2f} GB ({zip_path.stat().st_size:,} bytes)")
    print("=" * 80)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        infolist = zf.infolist()
        total_files = len(infolist)
        total_uncompressed = sum(info.file_size for info in infolist)
        print(f"Total entries: {total_files:,}")
        print(f"Total uncompressed size: {total_uncompressed / (1024**3):.2f} GB")
        
        ext_counter = Counter()
        dir_samples = set()
        sequences = set()
        xml_files = []
        sequence_frame_counts = defaultdict(int)
        
        for info in infolist:
            filename = info.filename
            parts = filename.strip("/").split("/")
            if len(parts) > 1:
                dir_samples.add(parts[0])
            
            ext = Path(filename).suffix.lower()
            if ext:
                ext_counter[ext] += 1
            
            if ext == ".xml":
                xml_files.append(filename)
            
            # Detect sequence IDs (MVI_XXXXX)
            m = re.search(r"(MVI_\d+)", filename)
            if m:
                seq_id = m.group(1)
                sequences.add(seq_id)
                if ext in [".jpg", ".jpeg", ".png"]:
                    sequence_frame_counts[seq_id] += 1

        print("\nTop-level directories/prefixes:", sorted(list(dir_samples))[:10])
        print("File extensions:", dict(ext_counter))
        print(f"Total sequence IDs detected: {len(sequences)}")
        print(f"Sample sequence IDs: {sorted(list(sequences))[:10]}")
        print(f"XML annotations count: {len(xml_files)}")
        if xml_files:
            print(f"Sample XML files: {xml_files[:10]}")
        
        print("\nTarget sequences check:")
        for t in TARGET_SEQUENCES:
            if t in sequences:
                frames = sequence_frame_counts.get(t, 0)
                # check if xml exists
                has_xml = any(t in x for x in xml_files)
                print(f"  - {t}: FOUND (Frames: {frames}, Has XML: {has_xml})")
            else:
                print(f"  - {t}: NOT in this archive")
    print("\n")
