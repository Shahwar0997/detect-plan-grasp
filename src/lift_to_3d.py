"""
lift_to_3d.py — back-project a 2D detection + depth into a 3D position.

Pinhole back-projection: a pixel (u,v) with metric depth Z maps to a camera-frame 3D point
    X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy,   Z = Z
Crop the depth under a detection box, keep the pixels near the object's depth (drop the
table/background/occluders), back-project them into a point cloud, and take the centroid as
the object's 3D position. Verified below against BOP ground-truth translations (cam_t_m2c).

Units: BOP depth PNG values * depth_scale = millimetres; cam_t_m2c is also in mm.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np


def back_project(depth_mm: np.ndarray, box, K, depth_band: float = 0.15) -> np.ndarray:
    """depth_mm: HxW (mm). box: (x1,y1,x2,y2) px. K: (fx,fy,cx,cy). -> Nx3 points (mm)."""
    fx, fy, cx, cy = K
    H, W = depth_mm.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3))
    z = depth_mm[y1:y2, x1:x2].astype(np.float32)
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    valid = z > 0
    if valid.sum() < 10:
        return np.empty((0, 3))
    # Robust object isolation: keep pixels within depth_band of the median depth in the box,
    # which drops the far background/table and near occluders, leaving the object's surface.
    med = np.median(z[valid])
    keep = valid & (np.abs(z - med) < depth_band * med)
    Z = z[keep]
    X = (us[keep] - cx) * Z / fx
    Y = (vs[keep] - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1)


def estimate_position(pts: np.ndarray):
    """Centroid of the object point cloud -> (X, Y, Z) mm in the camera frame."""
    return None if len(pts) == 0 else pts.mean(axis=0)


# --------------------------------------------------------------------------------------
# Verification against BOP ground truth (uses GT boxes to isolate the lift geometry).
# --------------------------------------------------------------------------------------
def _iter_gt(test_dir: Path, max_frames: int | None = None):
    n = 0
    for scene in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        cam = json.loads((scene / "scene_camera.json").read_text())
        gt = json.loads((scene / "scene_gt.json").read_text())
        info = json.loads((scene / "scene_gt_info.json").read_text())
        for img_id, objs in gt.items():
            depth_p = scene / "depth" / f"{int(img_id):06d}.png"
            if not depth_p.exists():
                continue
            K = cam[img_id]["cam_K"]
            K = (K[0], K[4], K[2], K[5])                       # fx, fy, cx, cy
            scale = cam[img_id]["depth_scale"]
            depth_mm = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
            for obj, meta in zip(objs, info[img_id]):
                if meta["visib_fract"] < 0.3:                  # need a decent view to localize
                    continue
                x, y, w, h = meta["bbox_visib"]
                if w <= 0 or h <= 0:
                    continue
                yield depth_mm, (x, y, x + w, y + h), K, np.array(obj["cam_t_m2c"])
            n += 1
            if max_frames and n >= max_frames:
                return


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parents[1]
    test_dir = repo / "data" / "test"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    errs, xy_errs, z_errs = [], [], []
    for depth_mm, box, K, gt_t in _iter_gt(test_dir, limit):
        pts = back_project(depth_mm, box, K)
        est = estimate_position(pts)
        if est is None:
            continue
        d = est - gt_t
        errs.append(np.linalg.norm(d))
        xy_errs.append(np.linalg.norm(d[:2]))
        z_errs.append(abs(d[2]))
    errs, xy_errs, z_errs = map(np.array, (errs, xy_errs, z_errs))
    print(f"objects localized: {len(errs)}")
    print(f"  3D position error  median {np.median(errs):6.1f} mm   mean {errs.mean():6.1f} mm")
    print(f"  lateral (XY) error median {np.median(xy_errs):6.1f} mm")
    print(f"  depth   (Z)  error median {np.median(z_errs):6.1f} mm")
