"""
make_panda_mobile.py — derive a mobile-base Panda from panda.xml.

Wraps the fixed arm base (link0) in a `mobile_base` body with slide-x, slide-y, and hinge-yaw
joints (+ position actuators), turning the fixed manipulator into a mobile manipulator that can
drive around a room. Arm/gripper joints and actuators keep their names, so the existing IK,
grasp, and control code is unchanged. Run: python sim/make_panda_mobile.py
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "franka" / "panda.xml"
DST = SRC.parent / "panda_mobile.xml"

BASE_OPEN = '''    <body name="mobile_base" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" damping="40"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="40"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="30"/>
      <geom name="base_geom" type="cylinder" size="0.14 0.05" pos="0 0 -0.02" rgba="0.16 0.17 0.22 1"/>
    <body name="link0" childclass="panda">'''

BASE_ACT = '''  <actuator>
    <position name="base_x" joint="base_x" kp="25000" dampratio="1" ctrlrange="-3 3"/>
    <position name="base_y" joint="base_y" kp="25000" dampratio="1" ctrlrange="-3 3"/>
    <position name="base_yaw" joint="base_yaw" kp="4000" dampratio="1" ctrlrange="-3.15 3.15"/>'''


def main():
    s = SRC.read_text()
    s = s.replace('    <body name="link0" childclass="panda">', BASE_OPEN, 1)
    s = s.replace('  </worldbody>', '    </body>\n  </worldbody>', 1)     # close mobile_base
    s = s.replace('  <actuator>', BASE_ACT, 1)
    s = s.replace(
        '<key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04" '
        'ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>',
        '<key name="home" qpos="0 0 0 0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04" '
        'ctrl="0 0 0 0 0 0 -1.57079 0 1.57079 -0.7853 255"/>')
    DST.write_text(s)
    print(f"wrote {DST.name} (mobile base: base_x, base_y, base_yaw)")


if __name__ == "__main__":
    main()
