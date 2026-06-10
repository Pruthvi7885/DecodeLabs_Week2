"""
test_pipeline.py
-----------------
Unit tests for the vision pipeline.
Run: python -m pytest tests/ -v
  or: python tests/test_pipeline.py
"""
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import cv2
import numpy as np
import math
import unittest
 
from vision_pipeline import (
    Config, phase1_preprocess, phase2_topology,
    phase3_tolerance_gate, compute_severity,
    InspectionResult, DefectPoint,
)
 
 
# ─── Helpers ───────────────────────────────────
def make_gear_image(size=256, broken_tooth=False, crack=False):
    """Generate a minimal synthetic gear image for testing."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    outer_r, inner_r = 90, 72
    num_teeth = 16
    tooth_h   = 14
    tooth_angle = 2 * math.pi / num_teeth
    half_tooth  = tooth_angle * 0.4
 
    pts = []
    for i in range(num_teeth):
        base_angle = i * tooth_angle
        a1  = base_angle - half_tooth
        pts.append([int(cx + inner_r * math.cos(a1)),
                    int(cy + inner_r * math.sin(a1))])
        tip_h = tooth_h * 0.2 if (broken_tooth and i == 3) else tooth_h
        tip_r = outer_r + tip_h
        for frac in [0.3, 0.5, 0.7]:
            a_tip = base_angle - half_tooth + frac * 2 * half_tooth
            pts.append([int(cx + tip_r * math.cos(a_tip)),
                        int(cy + tip_r * math.sin(a_tip))])
        a2  = base_angle + half_tooth
        pts.append([int(cx + inner_r * math.cos(a2)),
                    int(cy + inner_r * math.sin(a2))])
 
    contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [contour], 200)
 
    if crack:
        cv2.line(img, (cx + 60, cy + 20), (cx + 90, cy + 50), 50, 2)
 
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, img
 
 
class TestPhase1Preprocess(unittest.TestCase):
 
    def test_returns_required_keys(self):
        bgr, _ = make_gear_image()
        result  = phase1_preprocess(bgr)
        for key in ("gray", "blurred", "thresh", "thresh_val"):
            self.assertIn(key, result, f"Missing key '{key}' in Phase 1 output")
 
    def test_grayscale_is_single_channel(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        self.assertEqual(len(stages["gray"].shape), 2)
 
    def test_threshold_is_binary(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        unique  = set(stages["thresh"].flatten().tolist())
        self.assertTrue(unique.issubset({0, 255}),
                        f"Threshold not binary: unique values = {unique}")
 
    def test_thresh_val_in_range(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        self.assertGreater(stages["thresh_val"], 0)
        self.assertLessEqual(stages["thresh_val"], 255)
 
 
class TestPhase2Topology(unittest.TestCase):
 
    def test_detects_gear_contour(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        contour, hull, defects, metrics = phase2_topology(stages["thresh"])
        self.assertIsNotNone(contour, "Phase 2 should find the gear contour")
 
    def test_metrics_keys(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        _, _, _, metrics = phase2_topology(stages["thresh"])
        for key in ("area", "circularity", "diameter"):
            self.assertIn(key, metrics)
 
    def test_circularity_range(self):
        bgr, _ = make_gear_image()
        stages  = phase1_preprocess(bgr)
        _, _, _, metrics = phase2_topology(stages["thresh"])
        c = metrics["circularity"]
        self.assertGreater(c, 0.0)
        self.assertLessEqual(c, 1.0)
 
    def test_blank_image_returns_none(self):
        blank   = np.zeros((256, 256), dtype=np.uint8)
        contour, hull, defects, _ = phase2_topology(blank)
        self.assertIsNone(contour, "Blank image should return no contour")
 
 
class TestPhase3ToleranceGate(unittest.TestCase):
 
    def test_perfect_gear_no_defects(self):
        bgr, _ = make_gear_image(broken_tooth=False)
        stages  = phase1_preprocess(bgr)
        config  = Config()
        config.MIN_CONTOUR_AREA = 1000
        contour, hull, defect_data, _ = phase2_topology(stages["thresh"], config)
        if contour is None:
            self.skipTest("Contour not found for perfect gear test image")
        found = phase3_tolerance_gate(defect_data, contour, config)
        structural = [d for d in found if d.depth > config.DEFECT_THRESHOLD_MAX]
        self.assertEqual(len(structural), 0,
                         "Perfect gear should have zero structural defects")
 
    def test_defect_array_structure(self):
        # Synthetic defect_data array  (OpenCV format: N×1×4)
        defect_data = np.array([[[0, 5, 10, int(60 * 256)]]], dtype=np.int32)
        # Build a tiny contour with enough points
        pts   = []
        for i in range(30):
            a = i * 2 * math.pi / 30
            pts.append([[int(64 + 50 * math.cos(a)), int(64 + 50 * math.sin(a))]])
        gear_contour = np.array(pts, dtype=np.int32)
 
        config = Config()
        config.DEFECT_THRESHOLD_NORMAL = 10.0
        config.DEFECT_THRESHOLD_MAX    = 40.0
        found  = phase3_tolerance_gate(defect_data, gear_contour, config)
        self.assertIsInstance(found, list)
 
    def test_distance_scaling(self):
        """Verify the d_raw / 256.0 fix is applied."""
        raw_d = 12288   # = 48 * 256  → actual = 48.0 px
        pts   = []
        for i in range(30):
            a = i * 2 * math.pi / 30
            pts.append([[int(64 + 50 * math.cos(a)), int(64 + 50 * math.sin(a))]])
        gear_contour = np.array(pts, dtype=np.int32)
        defect_data  = np.array([[[0, 5, 10, raw_d]]], dtype=np.int32)
 
        config = Config()
        config.DEFECT_THRESHOLD_NORMAL = 10.0
        config.DEFECT_THRESHOLD_MAX    = 40.0
        found  = phase3_tolerance_gate(defect_data, gear_contour, config)
        if found:
            self.assertAlmostEqual(found[0].depth, 48.0, places=1)
 
 
class TestSeverityScoring(unittest.TestCase):
 
    def test_no_defects_zero_severity(self):
        self.assertEqual(compute_severity([]), 0.0)
 
    def test_severity_increases_with_depth(self):
        config = Config()
        d1 = DefectPoint((0,0),(1,1),(2,2), depth=50.0, defect_type="broken_tooth")
        d2 = DefectPoint((0,0),(1,1),(2,2), depth=100.0, defect_type="broken_tooth")
        s1 = compute_severity([d1], config)
        s2 = compute_severity([d2], config)
        self.assertGreater(s2, s1)
 
    def test_severity_capped_at_100(self):
        config = Config()
        defects= [DefectPoint((0,0),(1,1),(2,2), depth=9999, defect_type="x")
                  for _ in range(20)]
        self.assertLessEqual(compute_severity(defects, config), 100.0)
 
 
if __name__ == "__main__":
    unittest.main(verbosity=2)
 