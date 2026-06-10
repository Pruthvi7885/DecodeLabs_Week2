"""
batch_inspector.py
-------------------
Runs the full vision pipeline on the 20-image validation dataset.
Produces:
  - Annotated images in output/inspected/
  - JSON log in output/logs/inspection_log.json
  - Terminal accuracy report
  - Contact-sheet comparison image (output/contact_sheet.jpg)
 
Usage:
    python src/batch_inspector.py
    python src/batch_inspector.py --threshold 140
    python src/batch_inspector.py --debug
"""
 
import os
import sys
import json
import argparse
import time
import math
import cv2
import numpy as np
 
# Allow running from project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from vision_pipeline import Config, inspect_image
 
# ── Paths ──
ROOT        = os.path.join(os.path.dirname(__file__), "..")
PERFECT_DIR = os.path.join(ROOT, "dataset", "perfect")
DEFECT_DIR  = os.path.join(ROOT, "dataset", "defective")
OUT_DIR     = os.path.join(ROOT, "output", "inspected")
LOG_DIR     = os.path.join(ROOT, "output", "logs")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)
 
 
# ── ANSI colours for terminal ──
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
 
 
def collect_images():
    """Returns list of (path, ground_truth_label) tuples."""
    images = []
    for fname in sorted(os.listdir(PERFECT_DIR)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append((os.path.join(PERFECT_DIR, fname), "PASS"))
    for fname in sorted(os.listdir(DEFECT_DIR)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append((os.path.join(DEFECT_DIR, fname), "FAIL"))
    return images
 
 
def build_contact_sheet(results, out_path, cols=5, thumb_size=200):
    """
    Combine all annotated thumbnails into one contact sheet.
    Green border = correct classification, Red border = wrong.
    """
    thumbs = []
    for r in results:
        if r["annotated_img"] is not None:
            t = cv2.resize(r["annotated_img"], (thumb_size, thumb_size))
            correct = r["verdict"] == r["ground_truth"]
            border_col = (0, 180, 0) if correct else (0, 0, 220)
            t = cv2.copyMakeBorder(t, 4, 4, 4, 4,
                                   cv2.BORDER_CONSTANT, value=border_col)
            thumbs.append(t)
 
    if not thumbs:
        return
 
    rows = math.ceil(len(thumbs) / cols)
    thumb_h, thumb_w = thumbs[0].shape[:2]
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = t
 
    cv2.imwrite(out_path, sheet)
 
 
def run_batch(config: Config, debug: bool = False):
    images = collect_images()
    if not images:
        print(f"{RED}[ERROR] No images found. Run: python src/generate_dataset.py{RESET}")
        sys.exit(1)
 
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   DecodeLabs — Automated Gear Inspection      ║{RESET}")
    print(f"{BOLD}{CYAN}║   Batch: 2026 | Module: GEAR_INSPECTION_V2.0  ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}\n")
 
    log_records = []
    annotated_images = []
    tp = tn = fp = fn = 0
    total_ms = 0.0
 
    for idx, (path, gt) in enumerate(images, 1):
        result = inspect_image(path, config=config, save_debug=debug)
        correct = (result.verdict == gt)
 
        # Accumulate confusion matrix
        if   result.verdict == "PASS" and gt == "PASS": tn += 1
        elif result.verdict == "FAIL" and gt == "FAIL": tp += 1
        elif result.verdict == "FAIL" and gt == "PASS": fp += 1
        elif result.verdict == "PASS" and gt == "FAIL": fn += 1
 
        total_ms += result.processing_ms
 
        # Terminal row
        v_color = GREEN if result.verdict == "PASS" else RED
        c_mark  = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
        fname   = os.path.basename(path)
        defects = len(result.defects)
        print(f"  {idx:02d}. {fname:<30} │ GT:{gt:<4} │ "
              f"OUT:{v_color}{result.verdict:<4}{RESET} │ "
              f"SEV:{result.severity:>5.1f} │ "
              f"DEF:{defects} │ "
              f"{result.processing_ms:>5.1f}ms │ {c_mark}")
 
        # Save annotated image
        if result.annotated_img is not None:
            out_name = os.path.splitext(fname)[0] + "_inspected.jpg"
            out_path = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(out_path, result.annotated_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
 
        # JSON log record
        log_records.append({
            "id"          : idx,
            "filename"    : fname,
            "ground_truth": gt,
            "verdict"     : result.verdict,
            "correct"     : correct,
            "severity"    : result.severity,
            "defects"     : [
                {
                    "type" : d.defect_type,
                    "depth": d.depth,
                    "far_pt": list(d.far_pt),
                }
                for d in result.defects
            ],
            "gear_diameter"    : result.gear_diameter,
            "gear_circularity" : result.gear_circularity,
            "threshold_used"   : result.threshold_used,
            "processing_ms"    : round(result.processing_ms, 2),
        })
 
        annotated_images.append({
            "verdict"     : result.verdict,
            "ground_truth": gt,
            "annotated_img": result.annotated_img,
        })
 
    # ── Summary ──
    total      = tp + tn + fp + fn
    accuracy   = (tp + tn) / total * 100 if total else 0
    precision  = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall     = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_ms     = total_ms / total if total else 0
 
    print(f"\n{BOLD}{'─'*68}{RESET}")
    print(f"{BOLD}  INSPECTION REPORT — {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{'─'*68}")
    print(f"  Images Processed : {total}")
    print(f"  {GREEN}True  Positives (FAIL correctly caught) : {tp}{RESET}")
    print(f"  {GREEN}True  Negatives (PASS correctly passed) : {tn}{RESET}")
    print(f"  {RED}False Positives (good part rejected)    : {fp}{RESET}")
    print(f"  {RED}False Negatives (bad  part let through) : {fn}{RESET}")
    print(f"{'─'*68}")
    print(f"  {BOLD}Accuracy  : {accuracy:.1f}%{RESET}")
    print(f"  Precision : {precision:.1f}%")
    print(f"  Recall    : {recall:.1f}%")
    print(f"  F1 Score  : {f1:.1f}%")
    print(f"  Avg Proc  : {avg_ms:.1f} ms/frame")
    print(f"{'─'*68}\n")
 
    verdict_color = GREEN if accuracy >= 100 else YELLOW if accuracy >= 80 else RED
    print(f"  {verdict_color}{BOLD}SYSTEM STATUS: {'FACTORY READY ✓' if accuracy >= 100 else 'NEEDS CALIBRATION ⚠'}{RESET}\n")
 
    # ── Write JSON log ──
    log = {
        "timestamp"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config"     : {
            "blur_kernel"             : list(config.BLUR_KERNEL),
            "threshold_value"         : config.THRESHOLD_VALUE,
            "defect_threshold_normal" : config.DEFECT_THRESHOLD_NORMAL,
            "defect_threshold_max"    : config.DEFECT_THRESHOLD_MAX,
        },
        "summary"    : {
            "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "avg_ms": round(avg_ms, 2),
        },
        "results"    : log_records,
    }
    log_path = os.path.join(LOG_DIR, "inspection_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved → {log_path}")
 
    # ── Contact sheet ──
    import math
    sheet_path = os.path.join(ROOT, "output", "contact_sheet.jpg")
    build_contact_sheet(annotated_images, sheet_path)
    print(f"  Contact sheet → {sheet_path}\n")
 
 
def parse_args():
    p = argparse.ArgumentParser(description="DecodeLabs Gear Inspection — Batch Runner")
    p.add_argument("--threshold",    type=int,   default=0,
                   help="Binary threshold (0 = Otsu auto)")
    p.add_argument("--defect-max",   type=float, default=45.0,
                   help="Max defect depth in pixels (default 45)")
    p.add_argument("--defect-normal",type=float, default=18.0,
                   help="Normal tooth valley depth in pixels (default 18)")
    p.add_argument("--debug",        action="store_true",
                   help="Save debug stage images")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()
    cfg.THRESHOLD_VALUE      = args.threshold
    cfg.DEFECT_THRESHOLD_MAX    = args.defect_max
    cfg.DEFECT_THRESHOLD_NORMAL = args.defect_normal
    run_batch(cfg, debug=args.debug)