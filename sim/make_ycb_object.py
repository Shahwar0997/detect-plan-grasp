"""
make_ycb_object.py — convert a YCB model (BOP ycbv_models) into a MuJoCo mesh + texture.

    python sim/make_ycb_object.py --obj-id 4 --name soup

Needs data/models/obj_0000<ID>.ply (+ .png) from the ycbv_models download. Writes
sim/franka/assets/<name>.obj and <name>.png (texture downscaled so the repo stays lean;
still crisp enough for the detector). MuJoCo reads the .obj UVs and applies <name>.png via
the material declared in dpg_scene_ycb.xml.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import trimesh

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id", type=int, default=4, help="4 = 005_tomato_soup_can")
    ap.add_argument("--name", default="soup")
    ap.add_argument("--tex-size", type=int, default=1024)
    args = ap.parse_args()

    out = REPO / "sim/franka/assets"
    out.mkdir(parents=True, exist_ok=True)
    trimesh.load(str(REPO / f"data/models/obj_{args.obj_id:06d}.ply")).export(
        str(out / f"{args.name}.obj"))                      # .obj carries the UVs
    tex = cv2.imread(str(REPO / f"data/models/obj_{args.obj_id:06d}.png"))
    cv2.imwrite(str(out / f"{args.name}.png"),
                cv2.resize(tex, (args.tex_size, args.tex_size)))
    for extra in ("material.mtl", "material_0.png"):        # MuJoCo ignores the .mtl
        (out / extra).unlink(missing_ok=True)
    print(f"wrote {args.name}.obj + {args.name}.png ({args.tex_size}px)")


if __name__ == "__main__":
    main()
