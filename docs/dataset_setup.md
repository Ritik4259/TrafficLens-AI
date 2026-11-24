# UA-DETRAC Dataset Setup Guide

This guide describes how the **UA-DETRAC** (University at Albany DETection and tRACking) benchmark dataset is organized, verified, and extracted for TrafficLens-AI.

---

## 1. Dataset Overview

The UA-DETRAC benchmark is a real-world multi-object detection and tracking dataset captured across 24 traffic intersections in Beijing and Tianjin, China.
- **Total Sequences**: 100 video sequences (60 training sequences, 40 testing sequences).
- **Total Frames**: Over 140,000 frames captured at 25 fps with resolution $960 \times 540$ pixels.
- **Annotated Objects**: Over 8,250 unique vehicles producing 1.21 million bounding boxes.
- **Conditions**: Diverse environments categorized by weather and lighting (`night`, `cloudy`, `rainy`, `sunny`) and camera stability (`stable`, `unstable`).

---

## 2. Local Archive Structure

The local dataset archives are stored at:
- `C:\Users\bhard\Desktop\ua_detrac_training_set.zip` (5.24 GB compressed, 5.59 GB uncompressed, 83,976 files)
- `C:\Users\bhard\Desktop\ua_detrac_testing_set.zip` (3.96 GB compressed, 4.29 GB uncompressed, 56,460 files)

### Internal Archive Layout
Each sequence inside the archive is organized into a dedicated prefix folder:
```
ua_detrac_training_set.zip/
├── detrac_1_MVI_20011/
│   ├── image000001.jpg
│   ├── image000002.jpg
│   ├── ...
│   ├── image000664.jpg
│   ├── annotations.csv
│   └── annotations.zip (contains image000001.xml ... image000664.xml)
├── detrac_2_MVI_20012/
└── ... (60 sequences total: detrac_1_MVI_20011 to detrac_60_MVI_63563)
```

- **Frame Format**: High-quality JPEGs named `image%06d.jpg` ($960 \times 540$ pixels).
- **Annotations**:
  - `annotations.csv`: Contains columns `filename`, `width`, `height`, `class`, `xmin`, `ymin`, `xmax`, `ymax`, `confidence`, and `target_id`. Vehicles are annotated with `class="vehicle"` and their integer tracking ID; ignore regions are explicitly tagged with `class="ignored"` and `target_id=-1`.
  - `annotations.zip`: Contains per-frame Pascal VOC XML files (`image000001.xml` to `image000NNN.xml`).

---

## 3. Project Directory Layout

To avoid extracting 140,000 frames unnecessarily, our extraction script extracts **only** the 8 curated sequences into the project repository under `data/`:

```
TrafficLens-AI/
└── data/
    ├── raw/
    │   ├── images/
    │   │   ├── MVI_20011/
    │   │   │   ├── image000001.jpg
    │   │   │   ├── image000002.jpg
    │   │   │   └── ... (664 frames)
    │   │   ├── MVI_20012/ (936 frames)
    │   │   ├── MVI_39801/ (885 frames)
    │   │   ├── MVI_39811/ (1070 frames)
    │   │   ├── MVI_40131/ (1645 frames)
    │   │   ├── MVI_40141/ (1600 frames)
    │   │   ├── MVI_63521/ (2055 frames)
    │   │   └── MVI_63561/ (1285 frames)
    │   └── annotations/
    │       ├── MVI_20011.xml                # Canonical sequence-level XML
    │       ├── MVI_20011_annotations.csv    # Preserved original CSV
    │       ├── MVI_20011_voc_annotations.zip# Preserved original VOC XMLs
    │       └── ...
    └── processed/
        └── curated_subset/
            └── curated_manifest.json        # Curated split configuration
```

> [!NOTE]
> All files under `data/` are strictly excluded from version control via `.gitignore`. The original ZIP archives on Desktop remain completely untouched.

---

## 4. Curated 8-Sequence Benchmark Split

All 8 sequences in our provisional curated benchmark originate from `ua_detrac_training_set.zip`:

| Sequence ID | Archive Prefix | Condition | Camera State | Frames | Vehicle BBoxes | Ignored Boxes | Unique IDs | Split |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MVI_39801** | `detrac_17_MVI_39801` | `cloudy` | `stable` | 885 | 4,853 | 6,195 | 43 | `train` |
| **MVI_40141** | `detrac_24_MVI_40141` | `rainy` | `stable` | 1,600 | 6,222 | 11,200 | 33 | `train` |
| **MVI_63521** | `detrac_52_MVI_63521` | `sunny` | `stable` | 2,055 | 15,088 | 4,110 | 90 | `train` |
| **MVI_63561** | `detrac_58_MVI_63561` | `sunny` | `unstable` | 1,285 | 9,803 | 3,855 | 188 | `train` |
| **MVI_20012** | `detrac_2_MVI_20012` | `night` | `unstable` | 936 | 8,608 | 5,616 | 41 | `val` |
| **MVI_39811** | `detrac_18_MVI_39811` | `cloudy` | `stable` | 1,070 | 599 | 5,350 | 9 | `val` |
| **MVI_20011** | `detrac_1_MVI_20011` | `night` | `unstable` | 664 | 7,655 | 4,648 | 53 | `test` |
| **MVI_40131** | `detrac_23_MVI_40131` | `rainy` | `unstable` | 1,645 | 15,324 | 9,870 | 67 | `test` |
| **TOTAL** | — | — | — | **10,140** | **68,152** | **50,844** | **524** | — |

---

## 5. Execution Commands

### Extraction & Canonical XML Conversion
```bash
python experiments/src/dataset/extract_curated_dataset.py
```

### Full Real Dataset Verification
```bash
python experiments/src/dataset/verify_dataset.py
pytest -v
```
