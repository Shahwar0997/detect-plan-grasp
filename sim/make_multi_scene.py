"""
make_multi_scene.py — generate a MuJoCo scene with several YCB objects on the table.

Writes sim/franka/dpg_scene_multi.xml. Object bodies are named obj_<name> with a free joint,
so they can be repositioned at runtime (rearrangement). Meshes/textures come from
sim/make_ycb_object.py. Run:  python sim/make_multi_scene.py
"""
from __future__ import annotations
from pathlib import Path

# name (matches assets/<name>.obj + .png), initial (x, y), place height z, class label
OBJECTS = [
    ("soup",    0.42, -0.15, 0.36),
    ("mustard", 0.42,  0.14, 0.41),
    ("spam",    0.60, -0.14, 0.36),
    ("mug",     0.60,  0.15, 0.36),
]

HEAD = """<mujoco model="dpg multi-object scene">
  <include file="panda.xml"/>
  <statistic center="0.5 0 0.3" extent="1.1"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <global azimuth="140" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.6 0.8" rgb2="0.1 0.15 0.25"
             width="512" height="512"/>
    <material name="tablemat" rgba="0.55 0.42 0.30 1"/>
{assets}  </asset>
  <worldbody>
    <light pos="0.5 0 1.6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.3 0.35 0.4 1"/>
    <geom name="table" type="box" pos="0.5 0 0.15" size="0.28 0.38 0.15" material="tablemat"/>
{bodies}    <camera name="cam" pos="0.5 -0.6 0.9" xyaxes="1 0 0 0 0.75 0.66"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets, bodies = "", ""
    for name, x, y, z in OBJECTS:
        assets += (f'    <texture name="{name}_tex" type="2d" file="assets/{name}.png"/>\n'
                   f'    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.3"/>\n'
                   f'    <mesh name="{name}" file="{name}.obj" scale="0.001 0.001 0.001"/>\n')
        bodies += (f'    <body name="obj_{name}" pos="{x} {y} {z}">\n'
                   f'      <freejoint/>\n'
                   f'      <geom name="geom_{name}" type="mesh" mesh="{name}" material="{name}_mat" '
                   f'mass="0.1" friction="1.5 0.1 0.001"/>\n'
                   f'    </body>\n')
    out = repo / "sim/franka/dpg_scene_multi.xml"
    out.write_text(HEAD.format(assets=assets, bodies=bodies))
    print(f"wrote {out.name} with {len(OBJECTS)} objects:", [o[0] for o in OBJECTS])


if __name__ == "__main__":
    main()
