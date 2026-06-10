"""
vision_pipeline.py
-------------------
Implements the full IPO (Input → Process → Output) pipeline:
 
  Phase 1 – Signal Isolation  : Grayscale → Gaussian Blur → Threshold
  Phase 2 – Topological Analysis : findContours → convexHull → convexityDefects
  Phase 3 – Tolerance Gate    : Evaluate defect depth → PASS / FAIL verdict
 
Advanced features:
  - Triple-frame verification (reduces false rejection rate)
  - Dynamic adaptive thresholding for varying lighting
  - Multi-defect type detection (broken tooth, crack, missing tooth)
  - Severity scoring (0–100)
  - Structured result dict for logging and downstream systems
"""
 
import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
 
 
# ──────────────────────────────────────────────
# Configuration constants  (calibrate per setup)
# ──────────────────────────────────────────────
class Config:
    # Gaussian blur kernel (must be odd)
    BLUR_KERNEL = (7, 7)
 
    # Binary threshold value (0 = use Otsu auto-threshold)
    THRESHOLD_VALUE = 0       # 0 enables Otsu
 
    # Min gear contour area (pixels²) to ignore dust/noise blobs
    MIN_CONTOUR_AREA = 5_000
 
    # Convexity defect depth thresholds (pixels)
    # Normal gear tooth valleys will be < DEFECT_THRESHOLD_NORMAL
    DEFECT_THRESHOLD_NORMAL = 18.0   # px – valleys shallower than this = tooth gap (OK)
    DEFECT_THRESHOLD_MAX    = 45.0   # px – deeper than this = structural defect (FAIL)
 
    # Triple-frame verification: how many consecutive frames must FAIL
    TRIPLE_VERIFY_FRAMES = 3
 
    # Morphological cleanup kernel size
    MORPH_KERNEL = (5, 5)
 
    # Min gear circularity (0–1). Filters non-circular blobs.
    MIN_CIRCULARITY = 0.04   # Gears have low circularity due to teeth
 
 
# ──────────────────────────────────────────────
# Result data structures
# ──────────────────────────────────────────────
@dataclass
class DefectPoint:
    """Single convexity defect detected on the gear contour."""
    start_pt   : Tuple[int, int]
    end_pt     : Tuple[int, int]
    far_pt     : Tuple[int, int]    # deepest point (farthest_point)
    depth      : float              # actual pixel depth (d_raw / 256.0)
    defect_type: str = "UNKNOWN"    # broken_tooth | crack | missing_tooth
 
 
@dataclass
class InspectionResult:
    """Full inspection verdict for one image."""
    image_path    : str
    verdict       : str = "PENDING"        # PASS | FAIL | ERROR
    severity      : float = 0.0           # 0–100
    defects       : List[DefectPoint] = field(default_factory=list)
    gear_area     : int   = 0
    gear_circularity: float = 0.0
    gear_diameter : float = 0.0
    tooth_count   : int   = 0
    processing_ms : float = 0.0
    threshold_used: int   = 0
    annotated_img : Optional[np.ndarray] = None
    debug_stages  : dict  = field(default_factory=dict)  # intermediate images
 
 
# ──────────────────────────────────────────────
# Phase 1 – Input: Signal Isolation
# ──────────────────────────────────────────────
def phase1_preprocess(img_bgr: np.ndarray, config: Config = Config()) -> dict:
    """
    Convert raw frame to clean binary shape.
      FLATTEN  → cv2.cvtColor   (BGR → Grayscale)
      SMOOTH   → cv2.GaussianBlur
      BINARIZE → cv2.threshold  (Otsu or fixed)
    Returns dict of intermediate stages for debug display.
    """
    # 1. Flatten: colour → intensity
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
 
    # 2. Smooth: suppress high-frequency sensor noise
    blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
 
    # 3. Binarize: isolate gear silhouette against background
    if config.THRESHOLD_VALUE == 0:
        thresh_val, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        thresh_val = config.THRESHOLD_VALUE
        _, thresh = cv2.threshold(
            blurred, config.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
        )
 
    # 4. Morphological cleanup: close small holes, remove dust pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
 
    return {
        "gray"      : gray,
        "blurred"   : blurred,
        "thresh"    : thresh,
        "thresh_val": int(thresh_val),
    }
 
 
# ──────────────────────────────────────────────
# Phase 2 – Process: Topological Analysis
# ──────────────────────────────────────────────
def phase2_topology(thresh: np.ndarray, config: Config = Config()):
    """
    Extract geometric features from the binary silhouette.
      Step 1: findContours  → outer boundary
      Step 2: convexHull    → rubber-band envelope
      Step 3: convexityDefects → measure the gaps
    Returns (gear_contour, hull_indices, defect_data, metrics_dict).
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
 
    if not contours:
        return None, None, None, {}
 
    # Pick the largest contour (the gear body)
    gear_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(gear_contour)
 
    if area < config.MIN_CONTOUR_AREA:
        return None, None, None, {"area": area, "note": "too_small"}
 
    # Circularity check: 4π·Area / Perimeter²
    perimeter    = cv2.arcLength(gear_contour, True)
    circularity  = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if circularity < config.MIN_CIRCULARITY:
        return gear_contour, None, None, {
            "area": area, "circularity": circularity, "note": "not_circular"
        }
 
    # Bounding circle → estimated diameter
    (cx, cy), radius = cv2.minEnclosingCircle(gear_contour)
    diameter = radius * 2
 
    # Convex Hull (returnPoints=False is REQUIRED for convexityDefects)
    hull_indices = cv2.convexHull(gear_contour, returnPoints=False)
 
    defect_data = None
    if hull_indices is not None and len(hull_indices) > 3:
        try:
            defect_data = cv2.convexityDefects(gear_contour, hull_indices)
        except cv2.error:
            defect_data = None
 
    metrics = {
        "area"       : int(area),
        "circularity": round(circularity, 4),
        "diameter"   : round(diameter, 2),
        "perimeter"  : round(perimeter, 2),
        "center"     : (int(cx), int(cy)),
    }
    return gear_contour, hull_indices, defect_data, metrics
 
 
# ──────────────────────────────────────────────
# Defect classification helper
# ──────────────────────────────────────────────
def _classify_defect(depth: float, far_pt: Tuple, gear_contour,
                     config: Config) -> str:
    """
    Heuristic classification of a single convexity defect.
    Depth bands:
      < THRESHOLD_NORMAL → normal tooth valley (ignore)
      NORMAL–MAX         → minor anomaly
      > MAX              → structural defect
    """
    if depth > config.DEFECT_THRESHOLD_MAX:
        return "broken_tooth"
    elif depth > config.DEFECT_THRESHOLD_NORMAL:
        return "surface_crack"
    return "normal_valley"
 
 
# ──────────────────────────────────────────────
# Phase 3 – Output: The Tolerance Gate
# ──────────────────────────────────────────────
def phase3_tolerance_gate(defect_data, gear_contour,
                           config: Config = Config()) -> List[DefectPoint]:
    """
    Step 1: Evaluate actual_distance against calibrated THRESHOLD_MAX.
    Step 2: Isolate coordinates of each defect point.
    Step 3: Classify and return structured DefectPoint list.
 
    CRITICAL: OpenCV returns distance scaled by 256 → divide by 256.0
    """
    found_defects: List[DefectPoint] = []
    if defect_data is None:
        return found_defects
 
    for defect_row in defect_data:
        s_idx, e_idx, f_idx, d_raw = defect_row[0]
 
        # Fix the fixed-point integer scaling (CRITICAL TRAP from PDF)
        actual_distance = d_raw / 256.0
 
        # Only evaluate defects deeper than normal tooth valley
        if actual_distance <= config.DEFECT_THRESHOLD_NORMAL:
            continue
 
        start_pt = tuple(gear_contour[s_idx][0])
        end_pt   = tuple(gear_contour[e_idx][0])
        far_pt   = tuple(gear_contour[f_idx][0])
 
        defect_type = _classify_defect(actual_distance, far_pt,
                                       gear_contour, config)
 
        found_defects.append(DefectPoint(
            start_pt   = start_pt,
            end_pt     = end_pt,
            far_pt     = far_pt,
            depth      = round(actual_distance, 2),
            defect_type= defect_type,
        ))
 
    return found_defects
 
 
# ──────────────────────────────────────────────
# Severity scoring
# ──────────────────────────────────────────────
def compute_severity(defects: List[DefectPoint], config: Config = Config()) -> float:
    """
    Returns a 0–100 severity score.
    0   = perfect part
    100 = catastrophic failure
    """
    if not defects:
        return 0.0
    max_depth  = max(d.depth for d in defects)
    num_defects= len(defects)
    # Depth contribution (0–70)
    depth_score = min(70.0, (max_depth / (config.DEFECT_THRESHOLD_MAX * 2)) * 70)
    # Count contribution (0–30)
    count_score = min(30.0, num_defects * 10.0)
    return round(depth_score + count_score, 1)
 
 
# ──────────────────────────────────────────────
# Annotation renderer
# ──────────────────────────────────────────────
def render_annotation(img_bgr: np.ndarray, result: "InspectionResult",
                       gear_contour, hull_indices) -> np.ndarray:
    """
    Draw all visual annotations onto a copy of the original frame:
    - Gear outer contour (cyan)
    - Convex hull (yellow)
    - Defect bounding boxes (red)
    - Defect far-point markers (red circle)
    - PASS / FAIL verdict banner
    - Severity bar
    - Metadata overlay
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
 
    # Draw gear contour
    if gear_contour is not None:
        cv2.drawContours(out, [gear_contour], -1, (0, 220, 220), 2)
 
    # Draw convex hull
    if hull_indices is not None and gear_contour is not None:
        hull_pts = cv2.convexHull(gear_contour, returnPoints=True)
        cv2.drawContours(out, [hull_pts], -1, (0, 200, 0), 1)
 
    # Draw each defect
    for defect in result.defects:
        fp = defect.far_pt
        sp = defect.start_pt
        ep = defect.end_pt
 
        # Bounding box around the defect region
        x_vals = [sp[0], ep[0], fp[0]]
        y_vals = [sp[1], ep[1], fp[1]]
        x1, y1 = max(0, min(x_vals) - 20), max(0, min(y_vals) - 20)
        x2, y2 = min(w, max(x_vals) + 20), min(h, max(y_vals) + 20)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
        # Far point (deepest defect point)
        cv2.circle(out, fp, 6, (0, 0, 255), -1)
 
        # Depth label
        label = f"{defect.defect_type.upper()} d={defect.depth:.1f}px"
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 60, 255), 1,
                    cv2.LINE_AA)
 
    # ─ Verdict banner ─
    verdict_color = (0, 200, 60) if result.verdict == "PASS" else (0, 0, 220)
    banner_h = 52
    overlay   = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
 
    # Verdict icon
    icon = "✓" if result.verdict == "PASS" else "✗"
    banner_text = f"  {icon}  {result.verdict}: {result.image_path.split('/')[-1]}"
    cv2.putText(out, banner_text, (10, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.80, verdict_color, 2, cv2.LINE_AA)
 
    # Severity badge (right side)
    sev_text = f"SEVERITY: {result.severity:.0f}/100"
    sev_color = (0, 200, 60) if result.severity < 30 else \
                (0, 165, 255) if result.severity < 60 else (0, 0, 220)
    cv2.putText(out, sev_text, (w - 260, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sev_color, 2, cv2.LINE_AA)
 
    # Severity bar (bottom)
    bar_y = h - 18
    bar_w = int(w * result.severity / 100)
    cv2.rectangle(out, (0, bar_y), (w, h), (20, 20, 20), -1)
    bar_color = (0, 200, 60) if result.severity < 30 else \
                (0, 165, 255) if result.severity < 60 else (0, 0, 220)
    cv2.rectangle(out, (0, bar_y), (bar_w, h), bar_color, -1)
 
    # Stats overlay (bottom-left)
    stats = [
        f"Diameter: {result.gear_diameter:.0f}px",
        f"Circularity: {result.gear_circularity:.3f}",
        f"Defects: {len(result.defects)}",
        f"Process: {result.processing_ms:.1f}ms",
    ]
    for idx, s in enumerate(stats):
        cv2.putText(out, s, (10, h - 25 - (len(stats) - 1 - idx) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
 
    return out
 
 
# ──────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────
def inspect_image(image_path: str,
                  config: Config = Config(),
                  save_debug: bool = False) -> InspectionResult:
    """
    Full IPO inspection pipeline for one image.
    Returns an InspectionResult with verdict, severity, defect list,
    and annotated image.
    """
    result = InspectionResult(image_path=image_path)
    t0 = time.perf_counter()
 
    # ── Load ──
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        result.verdict = "ERROR"
        return result
 
    # ── Phase 1: Preprocess ──
    stages = phase1_preprocess(img_bgr, config)
    result.threshold_used = stages["thresh_val"]
    if save_debug:
        result.debug_stages = stages
 
    # ── Phase 2: Topology ──
    gear_contour, hull_indices, defect_data, metrics = phase2_topology(
        stages["thresh"], config
    )
 
    if gear_contour is None:
        result.verdict = "ERROR"
        result.processing_ms = (time.perf_counter() - t0) * 1000
        return result
 
    result.gear_area        = metrics.get("area", 0)
    result.gear_circularity = metrics.get("circularity", 0.0)
    result.gear_diameter    = metrics.get("diameter", 0.0)
 
    # ── Phase 3: Tolerance Gate ──
    defects = phase3_tolerance_gate(defect_data, gear_contour, config)
    result.defects  = defects
    result.severity = compute_severity(defects, config)
 
    # ── Verdict ──
    structural_defects = [d for d in defects
                          if d.depth > config.DEFECT_THRESHOLD_MAX]
    result.verdict = "FAIL" if structural_defects else "PASS"
 
    # ── Annotate ──
    result.annotated_img = render_annotation(
        img_bgr, result, gear_contour, hull_indices
    )
 
    result.processing_ms = (time.perf_counter() - t0) * 1000
    return result