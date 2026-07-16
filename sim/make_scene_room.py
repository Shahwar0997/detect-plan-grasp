"""
make_scene_room.py — a room with a MOBILE Panda that must navigate around obstacles to move an
object from a source bin to a target bin on the other side of the room.

The mobile base (panda_mobile.xml) starts near the source table; an obstacle wall blocks the
direct route, so the base must plan a 2D path (A* over an occupancy grid — see nav.py) around it
to reach the target table. Then the arm places the object. Bodies obj_<name> have free joints.
Run: python sim/make_scene_room.py
"""
from __future__ import annotations
from pathlib import Path

# world layout (metres). Base starts at origin (0,0).
SOURCE_TABLE = (0.75, -0.55)          # source table+bin center (arm-reachable from a nearby park)
TARGET_TABLE = (0.75,  1.35)          # target table, across the room
SOURCE_PARK = (0.28, -0.55)           # base pose to reach the source shelf (close, for a clean grasp)
TARGET_PARK = (0.28,  1.35)           # base pose to reach the target bin
OBSTACLES = [                          # (cx, cy, hx, hy) axis-aligned wall boxes
    (0.0, 0.4, 0.9, 0.06),            # long wall blocking the straight route
    (-0.9, 0.4, 0.06, 0.5),          # short return wall -> forces a detour to +x
]
# open source SHELF holds the target among distractors near the front edge (comfortable top-down
# reach); detection must pick the right one, and the open surface keeps grasps unobstructed
OBJECTS = [("soup", 0.52, -0.44, 0.36), ("mustard", 0.61, -0.66, 0.36),
           ("spam", 0.71, -0.48, 0.36)]


def _bin(name, x, y, z, mat, hx=0.14, hy=0.16):
    s = f'    <body name="{name}" pos="{x} {y} {z}">\n'
    s += (f'      <geom size="{hx} {hy} 0.02" type="box" group="0" friction="1 .005 .0001"/>\n'
          f'      <geom size="{hx} {hy} 0.02" type="box" contype="0" conaffinity="0" material="{mat}"/>\n')
    for wx, wy, sx, sy in [(0, hy, hx + .01, .01), (0, -hy, hx + .01, .01),
                           (hx, 0, .01, hy), (-hx, 0, .01, hy)]:
        s += (f'      <geom pos="{wx} {wy} 0.05" size="{sx} {sy} 0.05" type="box" group="0" friction="1 .005 .0001"/>\n'
              f'      <geom pos="{wx} {wy} 0.05" size="{sx} {sy} 0.05" type="box" contype="0" conaffinity="0" material="{mat}"/>\n')
    return s + '    </body>\n'


def main():
    repo = Path(__file__).resolve().parents[1]
    assets = ('    <texture type="skybox" builtin="gradient" rgb1="0.5 0.65 0.85" rgb2="0.1 0.15 0.25" width="512" height="512"/>\n'
              '    <texture name="ft" type="2d" file="assets/light-gray-floor-tile.png"/>\n'
              '    <material name="floormat" texture="ft" texrepeat="10 10" reflectance="0.05"/>\n'
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
                   f'mass="0.1" friction="3.0 0.3 0.01"/>\n    </body>\n')
    obst = ''.join(f'    <geom name="obst{i}" type="box" pos="{cx} {cy} 0.25" size="{hx} {hy} 0.25" '
                   f'rgba="0.7 0.25 0.2 1"/>\n' for i, (cx, cy, hx, hy) in enumerate(OBSTACLES))
    tables = ''.join(f'    <geom name="tbl{i}" type="box" pos="{tx} {ty} 0.15" size="0.3 0.3 0.15" material="tablemat"/>\n'
                     for i, (tx, ty) in enumerate([SOURCE_TABLE, TARGET_TABLE]))

    xml = f'''<mujoco model="dpg room (mobile manipulation)">
  <include file="panda_mobile.xml"/>
  <statistic center="0.4 0.4 0.3" extent="2.5"/>
  <visual><global offwidth="1280" offheight="960" azimuth="120" elevation="-35"/><headlight diffuse="0.7 0.7 0.7" ambient="0.5 0.5 0.5"/></visual>
  <asset>
{assets}  </asset>
  <worldbody>
    <light pos="0.4 0.4 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="6 6 0.05" material="floormat"/>
{tables}{obst}{_bin("bin2", *TARGET_TABLE, 0.33, "dark-wood")}{bodies}    <camera name="cam" pos="0.35 0.4 3.7" xyaxes="1 0 0 0 1 0.12"/>
    <camera name="srccam" pos="0.63 -1.15 0.9" xyaxes="1 0 0 0 0.75 0.66"/>
  </worldbody>
</mujoco>
'''
    out = repo / "sim/franka/dpg_scene_room.xml"
    out.write_text(xml)
    print(f"wrote {out.name}: mobile panda + source/target bins + {len(OBSTACLES)} obstacles")


if __name__ == "__main__":
    main()
