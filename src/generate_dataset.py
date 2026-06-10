"""
generate_dataset.py  (v2 - redesigned for reliable detection)
"""
import cv2
import numpy as np
import os
import math
import random
 
PERFECT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "perfect")
DEFECT_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset", "defective")
SIZE   = 512
CX, CY = 256, 256
 
 
def draw_gear_on_img(img, cx, cy, outer_r, inner_r, num_teeth,
                     tooth_h, broken_teeth=None, missing_teeth=None,
                     crack_positions=None, noise_level=6, brightness=1.0):
    broken_teeth    = broken_teeth    or []
    missing_teeth   = missing_teeth   or []
    crack_positions = crack_positions or []
 
    tooth_angle = 2 * math.pi / num_teeth
    half_tooth  = tooth_angle * 0.42
 
    # ── Draw gear body (disk) ──
    cv2.circle(img, (cx, cy), outer_r, 190, -1)
 
    # ── Cut tooth valleys ──
    for i in range(num_teeth):
        base = i * tooth_angle
        # Valley between tooth i and i+1
        valley_pts = []
        for frac in np.linspace(0, 1, 8):
            a = base + half_tooth + frac * (tooth_angle - 2 * half_tooth)
            r = inner_r
            valley_pts.append([int(cx + r * math.cos(a)),
                                int(cy + r * math.sin(a))])
        valley_pts.append([cx, cy])  # close to center
        vp = np.array(valley_pts, dtype=np.int32)
        cv2.fillPoly(img, [vp], 30)
 
    # ── Simulate missing or broken teeth ──
    for i in missing_teeth:
        base = i * tooth_angle
        # Remove entire tooth (make it look like a valley)
        pts = []
        for frac in np.linspace(-0.6, 0.6, 12):
            a = base + frac * tooth_angle
            r = inner_r - 8
            pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
        pts.append([cx, cy])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 28)
 
    for i in broken_teeth:
        base = i * tooth_angle
        # Remove top half of tooth
        pts = []
        for frac in np.linspace(-0.48, 0.48, 10):
            a   = base + frac * tooth_angle
            r   = outer_r - 2
            pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
        # outer circle closing
        for frac in np.linspace(0.48, -0.48, 8):
            a   = base + frac * tooth_angle
            r   = outer_r + tooth_h + 5
            pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 28)
 
    # ── Surface cracks ──
    for (angle, length) in crack_positions:
        r0 = inner_r + 15
        x1 = int(cx + r0 * math.cos(angle))
        y1 = int(cy + r0 * math.sin(angle))
        x2 = int(cx + (r0 + length) * math.cos(angle + 0.3))
        y2 = int(cy + (r0 + length) * math.sin(angle + 0.3))
        cv2.line(img, (x1, y1), (x2, y2), 35, 3)
        # Branch
        xb = int((x1+x2)//2 + 12*math.cos(angle+1.1))
        yb = int((y1+y2)//2 + 12*math.sin(angle+1.1))
        cv2.line(img, ((x1+x2)//2,(y1+y2)//2),(xb,yb), 38, 2)
 
    # ── Bore hole + hub ──
    cv2.circle(img, (cx,cy), int(outer_r * 0.22), 28, -1)
    cv2.circle(img, (cx,cy), int(outer_r * 0.22), 90, 2)
    cv2.circle(img, (cx,cy), int(outer_r * 0.35), 160, 3)
    # Keyways
    for k in range(4):
        a = k * math.pi / 2
        p1 = (int(cx + (outer_r*0.20)*math.cos(a)), int(cy + (outer_r*0.20)*math.sin(a)))
        p2 = (int(cx + (outer_r*0.24)*math.cos(a)), int(cy + (outer_r*0.24)*math.sin(a)))
        cv2.line(img, p1, p2, 55, 5)
 
    # ── Lighting / vignette ──
    Y, X = np.ogrid[:SIZE, :SIZE]
    dist  = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
    light = 1.0 - 0.45 * np.clip(dist / (SIZE * 0.55), 0, 1)
    out   = (img.astype(np.float32) * brightness * light)
 
    # ── Sensor noise ──
    noise = np.random.normal(0, noise_level, (SIZE, SIZE)).astype(np.float32)
    out   = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out
 
 
def generate_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(PERFECT_DIR, exist_ok=True)
    os.makedirs(DEFECT_DIR,  exist_ok=True)
 
    print("[GEN] Generating dataset v2 ...")
    for i in range(10):
        bg  = np.full((SIZE, SIZE), 20, dtype=np.uint8)
        img = draw_gear_on_img(bg, CX, CY,
                               outer_r=180, inner_r=148, num_teeth=24, tooth_h=32,
                               noise_level=random.randint(4,10),
                               brightness=random.uniform(0.88, 1.12))
        path = os.path.join(PERFECT_DIR, f"gear_perfect_{i+1:02d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [OK] Perfect  gear {i+1:02d}")
 
    defect_types = ["broken_tooth","surface_crack","missing_tooth"]
    for i in range(10):
        bg  = np.full((SIZE, SIZE), 20, dtype=np.uint8)
        dt  = defect_types[i % 3]
        kw  = {}
        if dt == "broken_tooth":
            kw["broken_teeth"]   = [random.randint(2, 21)]
        elif dt == "surface_crack":
            kw["crack_positions"]= [(random.uniform(0, 2*math.pi), random.randint(30,55))]
        else:
            kw["missing_teeth"]  = [random.randint(2, 21)]
 
        img = draw_gear_on_img(bg, CX, CY,
                               outer_r=180, inner_r=148, num_teeth=24, tooth_h=32,
                               noise_level=random.randint(5,12),
                               brightness=random.uniform(0.85, 1.10),
                               **kw)
        path = os.path.join(DEFECT_DIR, f"gear_defect_{i+1:02d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [!!] Defective gear {i+1:02d} ({dt})")
 
    print("\n[GEN] Done. 20 images written.")
 
 
if __name__ == "__main__":
    generate_all()