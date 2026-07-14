"""
prepare_ycb.py — Convert a YCB-Video (BOP format) split into YOLO detection labels.

BOP ships ground-truth 2D bounding boxes directly (in scene_gt_info.json), so this is a
pure JSON -> YOLO reformat: no mask processing needed. For each frame we read the object
ids (scene_gt.json) and their *visible* boxes (scene_gt_info.json), drop heavily occluded
instances, convert pixel boxes to normalized YOLO xywh, and symlink the RGB image.

Input (a BOP split dir, e.g. data/test):
    <split>/<scene_id>/rgb/<img_id>.png
                       /scene_gt.json        # obj_id + 6D pose, per image
                       /scene_gt_info.json    # bbox_visib [x,y,w,h] (px), visib_fract, per image
Output (a YOLO dataset):
    data/ycb_yolo/images/<name>/<scene>_<img>.png   # symlink to the source rgb
                  /labels/<name>/<scene>_<img>.txt   # one line per visible object
                  /data.yaml                          # 21 classes + split paths

Idempotent: re-running overwrites labels and refreshes symlinks. File-in / file-out.

Usage:
    python src/prepare_ycb.py --split-dir data/test --name val
    python src/prepare_ycb.py --split-dir data/train_pbr --name train   # (on Colab)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# YCB-Video's 21 objects, ordered by BOP obj_id 1..21. The integer class id we write is
# (obj_id - 1), so this list is the authoritative id -> name map for data.yaml.
YCB_CLASSES = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors",
    "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
]

# YCB-Video frames are all 640x480. We read it once and assert to catch surprises.
IMG_W, IMG_H = 640, 480


def convert_split(split_dir: Path, out_root: Path, name: str, min_visib: float) -> dict:
    """Convert one BOP split into YOLO images/labels under out_root. Returns stats."""
    img_dir = out_root / "images" / name
    lbl_dir = out_root / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
    n_images = n_boxes = n_dropped = 0
    per_class = [0] * len(YCB_CLASSES)

    for scene in scenes:
        gt = json.loads((scene / "scene_gt.json").read_text())
        info = json.loads((scene / "scene_gt_info.json").read_text())
        for img_id, objs in gt.items():
            infos = info[img_id]
            lines = []
            for obj, meta in zip(objs, infos):
                # Drop fully-occluded / invisible instances: their visible box is unusable.
                if meta["visib_fract"] < min_visib:
                    n_dropped += 1
                    continue
                x, y, w, h = meta["bbox_visib"]          # pixels, top-left origin
                if w <= 0 or h <= 0:                      # BOP marks absent boxes as -1
                    n_dropped += 1
                    continue
                cls = obj["obj_id"] - 1                    # 1..21 -> 0..20
                xc = (x + w / 2) / IMG_W                   # normalize to [0,1]
                yc = (y + h / 2) / IMG_H
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}")
                per_class[cls] += 1
                n_boxes += 1

            stem = f"{scene.name}_{int(img_id):06d}"
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
            # Symlink the source RGB (no copy): the YOLO dataset is a view over BOP.
            src_img = scene / "rgb" / f"{int(img_id):06d}.png"
            dst_img = img_dir / f"{stem}.png"
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
            n_images += 1

    return {"images": n_images, "boxes": n_boxes, "dropped": n_dropped,
            "per_class": per_class, "scenes": len(scenes)}


def write_data_yaml(out_root: Path) -> None:
    """Emit the Ultralytics manifest. Labels are found by swapping images/ -> labels/."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YCB_CLASSES))
    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(YCB_CLASSES)}\n"
        f"names:\n{names}\n"
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, help="BOP split dir (e.g. data/test)")
    ap.add_argument("--name", required=True, help="YOLO split name (train / val)")
    ap.add_argument("--out", default="data/ycb_yolo", help="output YOLO dataset root")
    ap.add_argument("--min-visib", type=float, default=0.1,
                    help="drop objects with visible fraction below this")
    args = ap.parse_args()

    split_dir = (repo / args.split_dir).resolve()
    out_root = (repo / args.out).resolve()
    stats = convert_split(split_dir, out_root, args.name, args.min_visib)
    write_data_yaml(out_root)

    print(f"[{args.name}] {stats['scenes']} scenes -> {stats['images']} images, "
          f"{stats['boxes']} boxes ({stats['dropped']} dropped, visib<{args.min_visib})")
    top = sorted(zip(YCB_CLASSES, stats["per_class"]), key=lambda t: -t[1])[:5]
    print("  top classes:", ", ".join(f"{n}={c}" for n, c in top))


if __name__ == "__main__":
    main()
