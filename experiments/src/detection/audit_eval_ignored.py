import json
from pathlib import Path
from experiments.src.dataset.parser import parse_detrac_xml

# Load test report
report_path = Path("experiments/outputs/evaluation/yolo11n_baseline_report.json")
with open(report_path) as f:
    data = json.load(f)

test_eval = data["test_evaluation"]
print("Total ignored detections reported:", test_eval["total_ignored_detections"])
print("Per-sequence test evaluation records:")
for s in test_eval["per_sequence"]:
    print(f"  {s['sequence_name']}: total_gts={s['total_gts']}, tp={s['tp_at_50']}, fp={s['fp_at_50']}, fn={s['fn_at_50']}, ignored_preds={s['ignored_preds']}")
