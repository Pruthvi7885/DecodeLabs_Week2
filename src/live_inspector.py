"""
live_inspector.py
------------------
Real-time video feed inspection using a webcam or video file.
Implements the triple-frame verification algorithm (reduces false
rejection by up to 28% vs. standalone camera logic — from PDF).
 
Usage:
    python src/live_inspector.py                  # webcam (index 0)
    python src/live_inspector.py --source 1       # second camera
    python src/live_inspector.py --source video.mp4
    python src/live_inspector.py --demo           # synthetic moving gear demo
 
Controls:
    Q / ESC   – quit
    S         – save current frame to output/
    P         – pause / resume
    D         – toggle debug overlay (shows grayscale / threshold stages)
    +/-       – increase / decrease defect threshold
"""
 
import cv2
import numpy as np
import os
import sys
import argparse
import time
import math
 
sys.path.insert(0, os.path.dirname(__file__))
from vision_pipeline import Config, inspect_image, phase1_preprocess, \
                             phase2_topology, phase3_tolerance_gate, \
                             compute_severity, render_annotation, InspectionResult
 
ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "output", "inspected")
os.makedirs(OUT_DIR, exist_ok=True)
 
 
# ──────────────────────────────────────────────
# Triple-Frame Verification Buffer
# ──────────────────────────────────────────────
class VerificationBuffer:
    """
    Implements the triple-frame verification algorithm.
    A FAIL is only confirmed when TRIPLE_VERIFY_FRAMES consecutive
    frames all return FAIL. This eliminates transient false positives
    caused by motion blur or lighting flicker.
    """
    def __init__(self, n: int = 3):
        self.n        = n
        self.history  = []           # stores "PASS" / "FAIL"
        self.confirmed= "PASS"       # the confirmed verdict
 
    def push(self, verdict: str) -> str:
        self.history.append(verdict)
        if len(self.history) > self.n:
            self.history.pop(0)
 
        if len(self.history) == self.n and all(v == "FAIL" for v in self.history):
            self.confirmed = "FAIL"
        elif "PASS" in self.history:
            self.confirmed = "PASS"
 
        return self.confirmed
 
    def reset(self):
        self.history  = []
        self.confirmed= "PASS"
 
 
# ──────────────────────────────────────────────
# Demo: Synthetic rotating gear frame generator
# ──────────────────────────────────────────────
def make_demo_frame(frame_idx: int, width=640, height=480):
    """Generate a synthetic rotating gear frame for demo mode."""
    img = np.full((height, width, 3), 22, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    outer_r, inner_r = 170, 140
    num_teeth = 24
    tooth_h   = 28
    tooth_angle = 2 * math.pi / num_teeth
    half_tooth  = tooth_angle * 0.35
 
    # Rotation offset
    rot = frame_idx * 0.04
 
    # Introduce a defect every 180 frames
    defect_tooth = 5 if (frame_idx // 180) % 2 == 1 else None
 
    pts = []
    for i in range(num_teeth):
        base_angle = i * tooth_angle + rot
        a1  = base_angle - half_tooth
        pts.append([int(cx + inner_r * math.cos(a1)),
                    int(cy + inner_r * math.sin(a1))])
        tip_h = tooth_h * 0.3 if i == defect_tooth else tooth_h
        tip_r = outer_r + tip_h
        for frac in [0.3, 0.5, 0.7]:
            a_tip = base_angle - half_tooth + frac * 2 * half_tooth
            pts.append([int(cx + tip_r * math.cos(a_tip)),
                        int(cy + tip_r * math.sin(a_tip))])
        a2  = base_angle + half_tooth
        pts.append([int(cx + inner_r * math.cos(a2)),
                    int(cy + inner_r * math.sin(a2))])
 
    contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [contour], color=(190, 190, 190))
    cv2.polylines(img, [contour], isClosed=True, color=(100, 100, 100), thickness=2)
    cv2.circle(img, (cx, cy), int(outer_r * 0.22), (30, 30, 30), -1)
    cv2.circle(img, (cx, cy), int(outer_r * 0.22), (80, 80, 80), 2)
 
    # Noise
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img
 
 
# ──────────────────────────────────────────────
# Run-time inspection on a raw BGR frame
# ──────────────────────────────────────────────
def inspect_frame(frame: np.ndarray, config: Config) -> InspectionResult:
    """Run the full IPO pipeline on a raw BGR frame (no file I/O)."""
    result = InspectionResult(image_path="<live>")
    t0 = time.perf_counter()
 
    stages = phase1_preprocess(frame, config)
    result.threshold_used = stages["thresh_val"]
 
    gear_contour, hull_indices, defect_data, metrics = phase2_topology(
        stages["thresh"], config
    )
 
    if gear_contour is None:
        result.verdict       = "NO GEAR"
        result.processing_ms = (time.perf_counter() - t0) * 1000
        result.debug_stages  = stages
        return result
 
    result.gear_area        = metrics.get("area", 0)
    result.gear_circularity = metrics.get("circularity", 0.0)
    result.gear_diameter    = metrics.get("diameter", 0.0)
 
    defects         = phase3_tolerance_gate(defect_data, gear_contour, config)
    result.defects  = defects
    result.severity = compute_severity(defects, config)
 
    structural = [d for d in defects if d.depth > config.DEFECT_THRESHOLD_MAX]
    result.verdict = "FAIL" if structural else "PASS"
 
    result.annotated_img = render_annotation(frame, result, gear_contour, hull_indices)
    result.debug_stages  = stages
    result.processing_ms = (time.perf_counter() - t0) * 1000
    return result
 
 
# ──────────────────────────────────────────────
# Debug stage overlay (4-panel)
# ──────────────────────────────────────────────
def make_debug_panel(stages: dict, annotated: np.ndarray) -> np.ndarray:
    """Create a 2×2 debug panel: original | gray | blurred | threshold."""
    target_h = annotated.shape[0] // 2
    target_w = annotated.shape[1] // 2
 
    def resize_and_bgr(img, label):
        if img is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (target_w, target_h))
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 220), 1, cv2.LINE_AA)
        return img
 
    gray    = resize_and_bgr(stages.get("gray"),    "GRAYSCALE")
    blurred = resize_and_bgr(stages.get("blurred"), "GAUSSIAN BLUR")
    thresh  = resize_and_bgr(stages.get("thresh"),  "THRESHOLD (BINARY)")
    annotated_small = cv2.resize(annotated, (target_w, target_h))
    cv2.putText(annotated_small, "ANNOTATED OUTPUT", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
 
    top    = np.hstack([gray, blurred])
    bottom = np.hstack([thresh, annotated_small])
    return np.vstack([top, bottom])
 
 
# ──────────────────────────────────────────────
# Main live loop
# ──────────────────────────────────────────────
def run_live(source, config: Config, demo: bool = False):
    cap = None
    if not demo:
        if isinstance(source, str) and not source.isdigit():
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(int(source) if isinstance(source, str) else source)
 
        if not demo and (cap is None or not cap.isOpened()):
            print(f"[WARN] Cannot open source '{source}'. Switching to demo mode.")
            demo = True
 
    verify_buf  = VerificationBuffer(config.TRIPLE_VERIFY_FRAMES)
    debug_mode  = False
    paused      = False
    frame_count = 0
    defect_log  = []       # rolling 60-frame defect count history
 
    # FPS tracking
    fps_counter = 0
    fps_ts      = time.time()
    fps_display = 0.0
 
    print("\n[LIVE] Starting inspection loop. Press Q to quit.")
 
    while True:
        if not paused:
            if demo:
                frame = make_demo_frame(frame_count)
                frame_count += 1
                time.sleep(0.033)   # ~30 fps synthetic
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[LIVE] End of stream.")
                    break
 
            result = inspect_frame(frame, config)
 
            # Triple-frame verification
            confirmed = verify_buf.push(result.verdict)
 
            # FPS
            fps_counter += 1
            if time.time() - fps_ts >= 1.0:
                fps_display = fps_counter / (time.time() - fps_ts)
                fps_counter = 0
                fps_ts      = time.time()
 
            # Rolling defect history (bar chart data)
            defect_log.append(len(result.defects))
            if len(defect_log) > 60:
                defect_log.pop(0)
 
        # ── Build display frame ──
        if debug_mode and result.debug_stages:
            display = make_debug_panel(
                result.debug_stages,
                result.annotated_img if result.annotated_img is not None else frame
            )
        else:
            display = result.annotated_img if result.annotated_img is not None else frame.copy()
 
        # Overlay: FPS + triple-verify state
        h, w = display.shape[:2]
        cv2.putText(display, f"FPS: {fps_display:.1f}", (w - 115, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
 
        # Triple-verify banner
        tv_color = (0, 180, 0) if confirmed == "PASS" else (0, 0, 220)
        tv_text  = f"[3-FRAME VERIFY]: {confirmed}"
        cv2.putText(display, tv_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, tv_color, 1, cv2.LINE_AA)
 
        if paused:
            cv2.putText(display, "PAUSED", (w//2 - 60, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)
 
        cv2.imshow("DecodeLabs | Gear Inspection System v2.0", display)
 
        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q / ESC
            break
        elif key == ord('s'):
            fname = os.path.join(OUT_DIR, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(fname, display)
            print(f"[SAVE] {fname}")
        elif key == ord('p'):
            paused = not paused
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[DEBUG] {'ON' if debug_mode else 'OFF'}")
        elif key == ord('+') or key == 43:
            config.DEFECT_THRESHOLD_MAX += 5
            print(f"[CFG] Defect threshold MAX → {config.DEFECT_THRESHOLD_MAX:.0f}px")
        elif key == ord('-') or key == 45:
            config.DEFECT_THRESHOLD_MAX = max(10, config.DEFECT_THRESHOLD_MAX - 5)
            print(f"[CFG] Defect threshold MAX → {config.DEFECT_THRESHOLD_MAX:.0f}px")
 
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[LIVE] Inspection session ended.")
 
 
def parse_args():
    p = argparse.ArgumentParser(description="DecodeLabs Live Gear Inspector")
    p.add_argument("--source",  default="0",
                   help="Camera index or video file path (default: 0)")
    p.add_argument("--demo",    action="store_true",
                   help="Run with synthetic rotating gear (no camera needed)")
    p.add_argument("--threshold",     type=int,   default=0)
    p.add_argument("--defect-max",    type=float, default=45.0)
    p.add_argument("--defect-normal", type=float, default=18.0)
    return p.parse_args()
 
 
if __name__ == "__main__":
    args   = parse_args()
    config = Config()
    config.THRESHOLD_VALUE       = args.threshold
    config.DEFECT_THRESHOLD_MAX  = args.defect_max
    config.DEFECT_THRESHOLD_NORMAL = args.defect_normal
    run_live(args.source, config, demo=args.demo)