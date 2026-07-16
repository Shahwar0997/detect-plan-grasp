"""
make_scene_bins.py — a pick-transport-place scene using robosuite's bins arena assets.

Two wood bins (robosuite's light-wood / dark-wood bins, with collision walls as obstacles)
sit in our Panda's workspace on a table. YCB objects start in bin 1; the task is to move one
to bin 2 — which requires lifting over the bin walls, i.e. a planned collision-free path.
Bodies obj_<name> have free joints (repositioned at runtime). Run: python sim/make_scene_bins.py
"""
from __future__ import annotations
from pathlib import Path

# YCB objects, starting inside bin 1 (near side, y < 0)
OBJECTS = [
    ("soup",  0.46, -0.24, 0.37),
    ("mug",   0.40, -0.33, 0.37),
    ("spam",  0.52, -0.16, 0.37),
]
BIN1 = (0.46, -0.24, 0.33)     # light-wood, source
BIN2 = (0.46,  0.24, 0.33)     # dark-wood, target
BIN_HX, BIN_HY = 0.14, 0.16    # bin inner half-extents


def _bin(name, x, y, z, mat):
    """robosuite-style bin: floor + 4 collision walls (the obstacles), no legs."""
    s = f'    <body name="{name}" pos="{x} {y} {z}">\n'
    s += (f'      <geom size="{BIN_HX} {BIN_HY} 0.02" type="box" group="0" friction="1 0.005 0.0001"/>\n'
          f'      <geom size="{BIN_HX} {BIN_HY} 0.02" type="box" contype="0" conaffinity="0" material="{mat}"/>\n')
    walls = [(0, BIN_HY, BIN_HX + 0.01, 0.01), (0, -BIN_HY, BIN_HX + 0.01, 0.01),
             (BIN_HX, 0, 0.01, BIN_HY), (-BIN_HX, 0, 0.01, BIN_HY)]
    for wx, wy, sx, sy in walls:
        s += (f'      <geom pos="{wx} {wy} 0.05" size="{sx} {sy} 0.05" type="box" group="0" friction="1 0.005 0.0001"/>\n'
              f'      <geom pos="{wx} {wy} 0.05" size="{sx} {sy} 0.05" type="box" contype="0" conaffinity="0" material="{mat}"/>\n')
    return s + '    </body>\n'


def main():
    repo = Path(__file__).resolve().parents[1]
    assets = ('    <texture type="skybox" builtin="gradient" rgb1="0.45 0.6 0.8" rgb2="0.1 0.15 0.25" width="512" height="512"/>\n'
              '    <texture name="floortex" type="2d" file="assets/light-gray-floor-tile.png"/>\n'
              '    <material name="floormat" texture="floortex" texrepeat="4 4" reflectance="0.05"/>\n'
              '    <texture name="lw" type="2d" file="assets/light-wood.png"/>\n'
              '    <texture name="dw" type="2d" file="assets/dark-wood.png"/>\n'
              '    <material name="light-wood" texture="lw" texrepeat="4 4"/>\n'
              '    <material name="dark-wood" texture="dw" texrepeat="3 3"/>\n'
              '    <material name="tablemat" rgba="0.5 0.4 0.32 1"/>\n')
    bodies = ''
    for name, x, y, z in OBJECTS:
        assets += (f'    <texture name="{name}_tex" type="2d" file="assets/{name}.png"/>\n'
                   f'    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.3"/>\n'
                   f'    <mesh name="{name}" file="{name}.obj" scale="0.001 0.001 0.001"/>\n')
        bodies += (f'    <body name="obj_{name}" pos="{x} {y} {z}">\n      <freejoint/>\n'
                   f'      <geom name="geom_{name}" type="mesh" mesh="{name}" material="{name}_mat" '
                   f'mass="0.1" friction="1.5 0.1 0.001"/>\n    </body>\n')

    xml = f'''<mujoco model="dpg bins scene (robosuite arena)">
  <include file="panda.xml"/>
  <statistic center="0.5 0 0.3" extent="1.2"/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <global azimuth="140" elevation="-25"/></visual>
  <asset>
{assets}  </asset>
  <worldbody>
    <light pos="0.5 0 1.6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floormat"/>
    <geom name="table" type="box" pos="0.5 0 0.15" size="0.36 0.44 0.15" material="tablemat"/>
    <!-- obstacle: a tall divider between the two bins -> the arm must plan a path over it -->
    <geom name="divider" type="box" pos="0.46 0 0.42" size="0.17 0.012 0.13" rgba="0.75 0.2 0.15 1"/>
{_bin("bin1", *BIN1, "light-wood")}{_bin("bin2", *BIN2, "dark-wood")}{bodies}    <camera name="cam" pos="1.25 0 0.82" xyaxes="0 1 0 -0.45 0 0.89"/>
  </worldbody>
</mujoco>
'''
    out = repo / "sim/franka/dpg_scene_bins.xml"
    out.write_text(xml)
    print(f"wrote {out.name}: 2 bins + {len(OBJECTS)} objects in bin1")


if __name__ == "__main__":
    main()
