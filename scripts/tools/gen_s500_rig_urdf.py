#!/usr/bin/env python3
"""Generate the S500 rig URDF: three S500 quadrotors carrying a disc on cables that can go slack.

Everything is one articulation rooted at the payload, so the hover environment drives it like the flycrane:

    load_link
      rope_<i>_ball_0_{x,y,z}          ball joint at the payload anchor
        rope_<i>_lower                 lower half of the cable, drawn from the anchor up
          rope_<i>_slide               prismatic, along the cable: 0 at full length, negative when shorter
            rope_<i>_upper             upper half of the cable, drawn from the drone's end down
              rope_<i>_ball_1_{x,y,z}  ball joint at the drone's cable mount
                Falcon<i>/mount_link   the tie point, on the rod between the landing gear's feet
                  Falcon<i>/base_link  the S500 body (fixed to the mount)
                    Falcon<i>/rotor_0..3   PX4 quad-X order

The cable is castor-main's `distance` cable: it has a maximum length and nothing else. PhysX's distance joint cannot
sit inside an articulation, so the same constraint is a telescoping link here: the slide joint's upper limit is the
cable's length, below it the link shortens freely. Taut, the slide sits on its limit and the joint's reaction force
is the cable tension; slack, the slide reads how much.

Links keep the ``Falcon<i>`` prefix the rest of this code base matches on; the airframe is an S500, drawn shape for
shape the way castor-main's build_vehicle_usd draws it.

Anchor i sits at ``angle_offset + (i - 1) * 120`` degrees around the payload, counter-clockwise seen from above. A
ball joint made of three hinges (x, then y, then z) locks up when its y hinge reaches 90 degrees, which is when the
cable lies along the joint frame's x axis. Each anchor frame is therefore yawed so that x is tangential to the
payload: a cable only lies along it when it is flat and wrapped around the rim, which flight does not produce,
whereas a cable pulled flat straight outward, which hard manoeuvres do produce, stays well conditioned. The spawn
pose is then two joint angles per cable, both on x hinges: ``+spawn_angle`` at the anchor tilts the cable outward,
``-spawn_angle`` at the mount levels the drone. They are written to ``s500_spawn.yaml``.

Needs only PyYAML, not Isaac Sim::

    python scripts/tools/gen_s500_rig_urdf.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR/flycastor_s500_data"
)

# castor-main's colours (castor_assets/usd_build.py), plus a heading mark for the payload
LOOKS = {
    "payload": "0.95 0.70 0.12 1",
    "mark": "0.1 0.1 0.1 1",
    "cable": "0.92 0.92 0.88 1",
    "carbon": "0.035 0.035 0.04 1",
    "nylon": "0.07 0.07 0.075 1",
    "pcb": "0.04 0.04 0.04 1",
    "motor": "0.06 0.06 0.065 1",
    "motor_cap": "0.75 0.06 0.06 1",
    "prop_ccw": "0.82 0.82 0.80 1",
    "prop_cw": "0.16 0.16 0.17 1",
    "foam": "0.03 0.03 0.03 1",
    "foam_ring": "0.70 0.05 0.05 1",
    "battery": "0.13 0.22 0.50 1",
    "avionics": "0.08 0.32 0.18 1",
    "gps_mast": "0.85 0.85 0.85 1",
    "rod": "0.55 0.55 0.58 1",
}
# URDF cylinders lie along z; these turn one onto another axis.
ALONG = {"z": (0, 0, 0), "x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0)}


def fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def inertial(mass: float, diag, xyz=(0, 0, 0)) -> str:
    return (
        f'    <inertial>\n      <origin rpy="0 0 0" xyz="{fmt(xyz)}"/>\n      <mass value="{mass:.6g}"/>\n'
        f'      <inertia ixx="{diag[0]:.6g}" ixy="0" ixz="0" iyy="{diag[1]:.6g}" iyz="0" izz="{diag[2]:.6g}"/>\n'
        "    </inertial>\n"
    )


def shape(tag: str, geometry: str, xyz=(0, 0, 0), rpy=(0, 0, 0), look: str | None = None) -> str:
    material = f'      <material name="{look}"/>\n' if look else ""
    return (
        f'    <{tag}>\n      <origin rpy="{fmt(rpy)}" xyz="{fmt(xyz)}"/>\n'
        f"      <geometry>{geometry}</geometry>\n{material}    </{tag}>\n"
    )


def box(size) -> str:
    return f'<box size="{fmt(size)}"/>'


def cylinder(radius: float, length: float) -> str:
    return f'<cylinder radius="{radius:.6g}" length="{length:.6g}"/>'


def link(name: str, body: str) -> str:
    return f'  <link name="{name}">\n{body}  </link>\n'


def joint(name, kind, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), axis=None, limit=None) -> str:
    out = f'  <joint name="{name}" type="{kind}">\n    <origin rpy="{fmt(rpy)}" xyz="{fmt(xyz)}"/>\n'
    out += f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
    if axis is not None:
        out += f'    <axis xyz="{fmt(axis)}"/>\n'
    if limit is not None:
        out += f'    <limit lower="{limit[0]:.6g}" upper="{limit[1]:.6g}" effort="1000" velocity="100"/>\n'
    return out + "  </joint>\n"


def ball_joint(prefix: str, parent: str, child: str, cable: dict, xyz=(0, 0, 0), rpy=(0, 0, 0)) -> str:
    """Three revolute axes with two near-massless links between them: URDF has no spherical joint."""
    tiny = inertial(cable["ball_link_mass"], [cable["ball_link_inertia"]] * 3)
    return (
        link(f"{prefix}_a", tiny)
        + link(f"{prefix}_b", tiny)
        + joint(f"{prefix}_x", "continuous", parent, f"{prefix}_a", xyz, rpy, (1, 0, 0))
        + joint(f"{prefix}_y", "continuous", f"{prefix}_a", f"{prefix}_b", axis=(0, 1, 0))
        + joint(f"{prefix}_z", "continuous", f"{prefix}_b", child, axis=(0, 0, 1))
    )


def payload_block(p: dict) -> str:
    r, h, m = p["radius"], p["height"], p["mass"]
    transverse = m * (3 * r * r + h * h) / 12.0
    geometry = cylinder(r, h)
    body = (
        inertial(m, (transverse, transverse, 0.5 * m * r * r))
        + shape("visual", geometry, look="payload")
        # a stripe along +x, so the payload's heading is visible
        + shape("visual", box((r, 0.02, 0.004)), xyz=(r / 2, 0, h / 2 + 0.002), look="mark")
        + shape("collision", geometry)
    )
    return link("load_link", body)


def cable_block(i: int, cable: dict, anchor_xyz, anchor_yaw: float) -> str:
    length, half = cable["length"], cable["length"] / 2
    mass = cable["mass"] / 2
    transverse = mass * half**2 / 12.0
    # Each half is drawn from its own end, so the two meet when the cable is taut and overlap when it is not.
    lower = inertial(mass, (transverse, transverse, 1e-6), xyz=(0, 0, half / 2)) + shape(
        "visual", cylinder(cable["radius"], half), xyz=(0, 0, half / 2), look="cable"
    )
    upper = inertial(mass, (transverse, transverse, 1e-6), xyz=(0, 0, -half / 2)) + shape(
        "visual", cylinder(cable["radius"], half), xyz=(0, 0, -half / 2), look="cable"
    )
    out = link(f"rope_{i}_lower", lower) + link(f"rope_{i}_upper", upper)
    out += ball_joint(f"rope_{i}_ball_0", "load_link", f"rope_{i}_lower", cable, anchor_xyz, (0, 0, anchor_yaw))
    out += joint(
        f"rope_{i}_slide", "prismatic", f"rope_{i}_lower", f"rope_{i}_upper", (0, 0, length), axis=(0, 0, 1),
        limit=(cable["min_length"] - length, 0.0),
    )
    out += ball_joint(f"rope_{i}_ball_1", f"rope_{i}_upper", f"Falcon{i}/mount_link", cable)
    return out


def drone_block(i: int, d: dict, anchor_yaw: float) -> str:
    """One S500, drawn the way castor-main's build_vehicle_usd draws it, shape for shape."""
    f, mo, lg, bm, rotor = d["frame"], d["motor"], d["landing_gear"], d["battery_mount"], d["rotor"]
    parts: list[str] = []

    def vis(look: str, geometry: str, xyz, rpy=(0, 0, 0)):
        parts.append(shape("visual", geometry, xyz, rpy, look))

    def col(geometry: str, xyz, rpy=(0, 0, 0)):
        parts.append(shape("collision", geometry, xyz, rpy))

    # z levels, as castor_assets.geometry.S500Geometry derives them
    arm_radius = f["wheelbase"] / 2
    z_arm_top, z_arm_bottom = f["arm_height"] / 2, -f["arm_height"] / 2
    z_top_plate = z_arm_top + f["plate_thickness"] / 2
    z_bottom_plate = z_arm_bottom - f["plate_thickness"] / 2
    z_underside = z_arm_bottom - f["plate_thickness"]
    z_motor_top = z_arm_top + mo["height"]
    z_pole_top = z_underside - lg["top_mount_height"]
    z_pole_bottom = z_pole_top - lg["pole_length"]
    tee_x, tee_y, tee_z = lg["tee_size"]
    z_skid = z_pole_bottom - tee_z + tee_y / 2

    # centre plates and arms
    px, py = f["plate_size"]
    vis("carbon", box((px, py, f["plate_thickness"])), (0, 0, z_top_plate))
    vis("pcb", box((px, py, f["plate_thickness"])), (0, 0, z_bottom_plate))
    col(box((px, py, z_top_plate - z_bottom_plate + f["plate_thickness"])), (0, 0, 0))
    arm_len = arm_radius - f["arm_root_radius"]
    mid = (f["arm_root_radius"] + arm_radius) / 2
    for x, y, _ in rotor["positions"]:
        ang = math.atan2(y, x)
        arm_xyz = (mid * math.cos(ang), mid * math.sin(ang), 0)
        vis("nylon", box((arm_len, f["arm_width"], f["arm_height"])), arm_xyz, (0, 0, ang))
        col(box((arm_len, f["arm_width"], f["arm_height"])), arm_xyz, (0, 0, ang))
        vis("nylon", cylinder(f["arm_width"] * 0.65, f["arm_height"]), (x, y, 0))
        vis("motor", cylinder(mo["diameter"] / 2, mo["height"]), (x, y, z_arm_top + mo["height"] / 2))
        vis("motor_cap", cylinder(mo["diameter"] / 2 * 0.7, 0.003), (x, y, z_motor_top + 0.0015))

    # landing gear: top mount, pole, tee, skid with foam, per side
    z_pole = (z_pole_top + z_pole_bottom) / 2
    foam_len = (lg["skid_length"] - tee_x) / 2 - 0.02
    for y in (lg["pole_spacing"] / 2, -lg["pole_spacing"] / 2):
        x = lg["pole_x"]
        vis("nylon", box((0.032, 0.030, lg["top_mount_height"])), (x, y, z_underside - lg["top_mount_height"] / 2))
        vis("carbon", cylinder(lg["pole_diameter"] / 2, lg["pole_length"]), (x, y, z_pole))
        vis("nylon", box(lg["tee_size"]), (x, y, z_pole_bottom - tee_z / 2))
        vis("carbon", cylinder(lg["skid_diameter"] / 2, lg["skid_length"]), (x, y, z_skid), ALONG["x"])
        for sx in (-1, 1):
            cx = x + sx * (tee_x / 2 + 0.005 + foam_len / 2)
            vis("foam", cylinder(lg["foam_diameter"] / 2, foam_len), (cx, y, z_skid), ALONG["x"])
            for j in range(3):
                ring_xyz = (cx + (j - 1) * foam_len / 3, y, z_skid)
                vis("foam_ring", cylinder(lg["foam_diameter"] / 2 + 0.0005, 0.004), ring_xyz, ALONG["x"])
        col(cylinder(lg["pole_diameter"] / 2, lg["pole_length"]), (x, y, z_pole))
        skid_radius = max(lg["foam_diameter"], lg["skid_diameter"]) / 2
        col(cylinder(skid_radius, lg["skid_length"]), (x, y, z_skid), ALONG["x"])

    # battery mount: two rails under the frame, on four brackets
    z_rail = z_underside - bm["drop"]
    bracket_h = bm["drop"] + bm["rail_diameter"] / 2
    for y in (bm["rail_spacing"] / 2, -bm["rail_spacing"] / 2):
        vis("carbon", cylinder(bm["rail_diameter"] / 2, bm["rail_length"]), (0, y, z_rail), ALONG["x"])
        for sx in (-1, 1):
            vis("nylon", box((0.012, 0.016, bracket_h)), (sx * 0.055, y, z_underside - bracket_h / 2))

    # battery, avionics, GPS on its mast
    for name, item in d["boxes"].items():
        vis(name, box(item["size"]), item["pos"])
    battery, gps = d["boxes"]["battery"], d["boxes"]["gps_mast"]
    col(box(battery["size"]), battery["pos"])
    mast_h = gps["pos"][2] - gps["size"][2] / 2 - z_top_plate
    vis("carbon", cylinder(0.005, mast_h), (gps["pos"][0], gps["pos"][1], z_top_plate + mast_h / 2))

    # the cross rod the cable ties to
    rod_length = lg["pole_spacing"] + lg["pole_diameter"]
    vis("rod", cylinder(d["mount_rod_diameter"] / 2, rod_length), (lg["pole_x"], 0, d["mount_offset_z"]), ALONG["y"])

    body = inertial(d["body_mass"], d["inertia"], d["com"]) + "".join(parts)
    mount = inertial(1e-4, (1e-6, 1e-6, 1e-6))

    name = f"Falcon{i}"
    out = link(f"{name}/mount_link", mount) + link(f"{name}/base_link", body)
    # The mount frame has the anchor frame's yaw, a quarter turn past the anchor's azimuth: turn it back to face
    # outward, or all the way back to the payload's +x for "forward".
    yaw = -math.pi / 2 if d["heading"] == "outward" else -math.pi / 2 - anchor_yaw
    out += joint(
        f"{name}/mount_joint", "fixed", f"{name}/mount_link", f"{name}/base_link", (0, 0, -d["mount_offset_z"]), (0, 0, yaw)
    )

    # rotors: a hub and two blades, pitched 10 degrees the way the propeller turns
    blade_len = rotor["prop_radius"] - 0.008
    for k, pos in enumerate(rotor["positions"]):
        ccw = rotor["spin"][k] < 0
        look = "prop_ccw" if ccw else "prop_cw"
        twist = math.radians(10.0) * (1 if ccw else -1)
        prop = inertial(rotor["prop_mass"], rotor["prop_inertia"])
        prop += shape("visual", cylinder(0.008, 0.010), look="motor")
        for sx, blade_yaw in ((1, 0.0), (-1, math.pi)):
            blade_xyz = (sx * (0.008 + blade_len / 2), 0, 0)
            prop += shape("visual", box((blade_len, 0.022, 0.003)), blade_xyz, (twist, 0, blade_yaw), look)
        out += link(f"{name}/rotor_{k}", prop)
        out += joint(f"{name}/rotor_{k}_joint", "continuous", f"{name}/base_link", f"{name}/rotor_{k}", pos, axis=(0, 0, 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DATA_DIR / "s500_rig.yaml")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "flycastor_s500.urdf")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    n, p, cable, d = cfg["num_drones"], cfg["payload"], cfg["cable"], cfg["drone"]

    urdf = '<?xml version="1.0" ?>\n<!-- Generated by scripts/tools/gen_s500_rig_urdf.py. Do not edit by hand. -->\n'
    urdf += '<robot name="flycastor_s500">\n'
    for look, rgba in LOOKS.items():
        urdf += f'  <material name="{look}"><color rgba="{rgba}"/></material>\n'
    urdf += payload_block(p)
    spawn = {}
    theta = math.radians(cable["spawn_angle_deg"])
    for i in range(1, n + 1):
        phi = math.radians(p["angle_offset_deg"]) + 2 * math.pi * (i - 1) / n
        anchor = (p["anchor_radius"] * math.cos(phi), p["anchor_radius"] * math.sin(phi), p["anchor_height"])
        urdf += cable_block(i, cable, anchor, phi + math.pi / 2) + drone_block(i, d, phi)
        spawn[f"rope_{i}_ball_0_x"] = round(theta, 6)
        spawn[f"rope_{i}_ball_1_x"] = round(-theta, 6)
    urdf += "</robot>\n"

    args.out.write_text(urdf)
    spawn_path = args.out.with_name("s500_spawn.yaml")
    spawn_path.write_text(
        "# Generated by scripts/tools/gen_s500_rig_urdf.py: joint angles that put the rig in its spawn pose.\n"
        + yaml.safe_dump({"joint_pos": spawn}, sort_keys=True)
    )

    radius = p["anchor_radius"] + cable["length"] * math.sin(theta)
    height = p["anchor_height"] + cable["length"] * math.cos(theta) - d["mount_offset_z"]
    weight = (d["body_mass"] + 4 * d["rotor"]["prop_mass"] + (p["mass"] + n * cable["mass"]) / n) * 9.8066
    tension = p["mass"] * 9.8066 / n / math.cos(theta)
    print(f"wrote {args.out}\nwrote {spawn_path}")
    print(f"  {n} drones, {urdf.count('<link ')} links, {urdf.count('<joint ')} joints")
    print(f"  drones spawn {radius:.3f} m from the payload axis, {height:.3f} m above its centre")
    for i in range(1, n + 1):
        print(f"  Falcon{i} (one-hot slot {i - 1}): anchor at {p['angle_offset_deg'] + 360.0 * (i - 1) / n:6.1f} deg")
    print(f"  hover thrust per motor {weight / 4:.3f} N, cable tension at rest {tension:.2f} N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
