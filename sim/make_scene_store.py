"""
make_scene_store.py — a larger "store" scene: 4 labeled shelves (A, B, C, D) around an open centre
that the mobile Panda navigates. Each shelf holds a few YCB objects (some duplicated across shelves)
and has its own oblique shelf-camera for detection. The simulation knows every shelf's location
(SHELVES registry) and a park pose in front of it.

This is Step 1 of the prompt-driven store demo: build + verify the world (renders, per-shelf
detection). The LLM task parsing and the end-to-end run come in later steps.
Run: python sim/make_scene_store.py
"""
from __future__ import annotations
from pathlib import Path

# ---- world layout (metres) -------------------------------------------------------------------
ROBOT_START = (0.0, 0.0)
SHELF_HX, SHELF_HY, SHELF_TOP = 0.38, 0.20, 0.40      # shelf-top half-extents and height
OBJ_Z = SHELF_TOP + 0.04                              # objects rest just above the shelf top
PARK_GAP = 0.72                                       # how far in front of a shelf the robot parks

# shelf: (cx, cy, cam_side)  cam_side = -1 -> camera/robot on the -y side (looks +y);
#                                    +1 -> on the +y side (looks -y). Objects face the camera.
SHELVES = {
    "A": (-1.15,  1.30, -1),      # back-left
    "B": ( 1.15,  1.30, -1),      # back-right
    "C": (-1.15, -1.30,  1),      # front-left
    "D": ( 1.15, -1.30,  1),      # front-right
}

# objects on each shelf: (mesh, shelf, dx)  -> placed along the shelf front at (cx+dx, cy + face*0.04)
OBJECTS = [
    ("soup", "A", -0.20), ("mustard", "A", 0.02), ("spam", "A", 0.22),
    ("mug", "B", -0.20), ("soup", "B", 0.02), ("bleach", "B", 0.22),
    ("cracker", "C", -0.20), ("gelatin", "C", 0.04), ("mustard", "C", 0.22),
    ("chef", "D", -0.20), ("spam", "D", 0.04), ("banana", "D", 0.22),
]


def shelf_park(name):
    cx, cy, side = SHELVES[name]
    return (cx, cy + side * PARK_GAP)                # robot parks on the camera side of the shelf


def shelf_cam_xml(name):
    cx, cy, side = SHELVES[name]
    # oblique, label-on view (matches the detector's training distribution). For a shelf whose
    # camera sits on the +y side (side=+1) the whole frame is the mirror of the -y case (rotate the
    # camera 180 deg about vertical), so it still looks AT the shelf and down.
    y = cy + side * 0.62
    return (f'    <camera name="cam_{name}" pos="{cx:.3f} {y:.3f} 0.90" '
            f'xyaxes="{-side} 0 0 0 {-0.75 * side:.2f} 0.66"/>\n')


def main():
    repo = Path(__file__).resolve().parents[1]
    meshes = sorted({m for m, *_ in OBJECTS})
    assets = ('    <texture type="skybox" builtin="gradient" rgb1="0.5 0.65 0.85" rgb2="0.1 0.15 0.25" width="512" height="512"/>\n'
              '    <texture name="ft" type="2d" file="assets/light-gray-floor-tile.png"/>\n'
              '    <material name="floormat" texture="ft" texrepeat="14 14" reflectance="0.05"/>\n'
              '    <texture name="lw" type="2d" file="assets/light-wood.png"/>\n'
              '    <material name="shelfmat" texture="lw" texrepeat="3 3"/>\n'
              '    <material name="labelmat" rgba="0.15 0.17 0.22 1"/>\n')
    for m in meshes:
        assets += (f'    <texture name="{m}_tex" type="2d" file="assets/{m}.png"/>\n'
                   f'    <material name="{m}_mat" texture="{m}_tex" specular="0.2" shininess="0.3"/>\n'
                   f'    <mesh name="{m}" file="{m}.obj" scale="0.001 0.001 0.001"/>\n')

    # shelves (raised platforms) + a label plaque on the camera-facing side
    shelves = ''
    for name, (cx, cy, side) in SHELVES.items():
        shelves += (f'    <geom name="shelf_{name}" type="box" pos="{cx} {cy} {SHELF_TOP/2:.3f}" '
                    f'size="{SHELF_HX} {SHELF_HY} {SHELF_TOP/2:.3f}" material="shelfmat"/>\n'
                    f'    <geom name="label_{name}" type="box" pos="{cx} {cy + side*(SHELF_HY+0.01):.3f} 0.30" '
                    f'size="0.09 0.005 0.06" material="labelmat"/>\n')

    # objects (unique body names since meshes repeat across shelves), oriented to face the camera
    bodies = ''
    for i, (mesh, shelf, dx) in enumerate(OBJECTS):
        cx, cy, side = SHELVES[shelf]
        x, y = cx + dx, cy + side * 0.04
        yaw = 0.0 if side < 0 else 3.14159             # face the camera (side +1 -> rotate 180)
        qw, qz = (1.0, 0.0) if side < 0 else (0.0, 1.0)
        bodies += (f'    <body name="obj_{shelf}_{mesh}_{i}" pos="{x:.3f} {y:.3f} {OBJ_Z}" '
                   f'quat="{qw} 0 0 {qz}">\n      <freejoint/>\n'
                   f'      <geom name="g_{shelf}_{mesh}_{i}" type="mesh" mesh="{mesh}" material="{mesh}_mat" '
                   f'mass="0.1" condim="6" friction="1.5 0.1 0.001"/>\n    </body>\n')

    cams = ''.join(shelf_cam_xml(n) for n in SHELVES)
    xml = f'''<mujoco model="dpg store (prompt-driven mobile manipulation)">
  <include file="panda_mobile.xml"/>
  <statistic center="0 0 0.3" extent="3.2"/>
  <visual><global offwidth="1280" offheight="960" azimuth="120" elevation="-40"/><headlight diffuse="0.7 0.7 0.7" ambient="0.5 0.5 0.5"/></visual>
  <asset>
{assets}  </asset>
  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="8 8 0.05" material="floormat"/>
{shelves}{bodies}{cams}    <camera name="overview" pos="0 -0.2 4.2" xyaxes="1 0 0 0 1 0.1"/>
  </worldbody>
</mujoco>
'''
    out = repo / "sim/franka/dpg_scene_store.xml"
    out.write_text(xml)
    print(f"wrote {out.name}: {len(SHELVES)} shelves, {len(OBJECTS)} objects "
          f"({len(meshes)} unique meshes), park poses + shelf cameras")


if __name__ == "__main__":
    main()
