#!/usr/bin/env python3
"""Generate an N-MAV flycrane URDF by cloning the shipped flypent structure.

Rather than emitting URDF from templates (~380 lines per drone, easy to get
subtly wrong), this reads a known-good flycrane-family URDF, extracts the
Falcon-1 block and the rope-1 chain as templates, and stamps out N copies with
renumbered link/joint names and the anchor points you specify.

Everything else - the payload, its odometry frame, the mesh filenames - is
carried through untouched, so the .dae meshes are reused verbatim and no 3D
modelling is required.

    python gen_flycrane_urdf.py flypent.urdf flyhex.urdf --n 6

Anchors default to a symmetric 3+3 layout on the payload's top face for N=6;
pass --anchors to override. The splay (rpy) on each anchor is what keeps the
MAVs apart at spawn - see --splay-x / --splay-y.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import xml.etree.ElementTree as ET

FALCON_RE = re.compile(r"^Falcon(\d+)/")
ROPE_RE = re.compile(r"^rope_(\d+)_")


def default_anchors(n: int, splay_x: float, splay_y: float,
                    half_x: float = 0.27, half_y: float = 0.22, z: float = 0.14):
    """Anchor points on the payload top face, paired left/right along y.

    N=6 gives two rows of three. Odd N puts the last anchor on the -x centre
    line, matching how flypent is laid out.
    """
    xs: list[float]
    rows = (n + 1) // 2
    if rows == 1:
        xs = [0.0]
    else:
        xs = [half_x - 2 * half_x * k / (rows - 1) for k in range(rows)]

    out = []
    for k, xv in enumerate(xs):
        sx = 0 if abs(xv) < 1e-9 else (1 if xv > 0 else -1)
        for sy in (1, -1):
            if len(out) == n:
                break
            if len(out) == n - 1 and n % 2 == 1:
                # last anchor of an odd team: centre it on y
                out.append(((xv, 0.0, z), (0.0, splay_x * sx, 0.0)))
                break
            out.append(((xv, sy * half_y, z), (-splay_y * sy, splay_x * sx, 0.0)))
    return out[:n]


def collect(root: ET.Element, pattern: re.Pattern, index: int):
    """Every link/joint whose own name, parent or child matches pattern with `index`."""
    keep = []
    for el in list(root):
        if el.tag not in ("link", "joint"):
            continue
        name = el.get("name", "")
        m = pattern.match(name)
        if m and int(m.group(1)) == index:
            keep.append(el)
    return keep


def renumber(el: ET.Element, old: int, new: int) -> ET.Element:
    """Deep-copy an element, rewriting Falcon<old>/ and rope_<old>_ to <new>."""
    c = copy.deepcopy(el)
    def fix(s: str) -> str:
        s = re.sub(rf"^Falcon{old}/", f"Falcon{new}/", s)
        s = re.sub(rf"^rope_{old}_", f"rope_{new}_", s)
        return s
    if c.get("name"):
        c.set("name", fix(c.get("name")))
    for tag in ("parent", "child"):
        sub = c.find(tag)
        if sub is not None and sub.get("link"):
            sub.set("link", fix(sub.get("link")))
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="a shipped flycrane-family URDF to use as the template")
    ap.add_argument("outfile")
    ap.add_argument("--n", type=int, required=True, help="number of MAVs")
    ap.add_argument("--splay-x", type=float, default=0.9, help="anchor pitch, rad (separates rows along x)")
    ap.add_argument("--splay-y", type=float, default=0.5, help="anchor roll, rad (separates the two y columns)")
    ap.add_argument("--robot-name", default=None)
    ap.add_argument("--anchors", default=None,
                    help='override, e.g. "0.27,0.22,0.14,-0.5,0.9,0; 0,0.22,0.14,-0.5,0,0; ..."')
    args = ap.parse_args()

    tree = ET.parse(args.infile)
    root = tree.getroot()

    falcon_ids = sorted({int(m.group(1)) for el in root
                         for m in [FALCON_RE.match(el.get("name", "") or "")] if m})
    rope_ids = sorted({int(m.group(1)) for el in root
                       for m in [ROPE_RE.match(el.get("name", "") or "")] if m})
    if not falcon_ids or not rope_ids:
        raise SystemExit("template has no Falcon*/ or rope_*_ elements")
    src = min(falcon_ids)
    print(f"template: {args.infile}")
    print(f"  found {len(falcon_ids)} MAVs, {len(rope_ids)} cables; cloning MAV {src}")

    falcon_tpl = collect(root, FALCON_RE, src)
    rope_tpl = collect(root, ROPE_RE, src)
    if not falcon_tpl or not rope_tpl:
        raise SystemExit("could not extract templates")
    seg = len([e for e in rope_tpl
               if e.tag == "link" and re.match(rf"^rope_{src}_link(_\d+)?$", e.get("name", ""))])
    print(f"  MAV block: {len(falcon_tpl)} elements | cable: {len(rope_tpl)} elements, {seg} segment(s)")

    if args.anchors:
        anchors = []
        for chunk in args.anchors.split(";"):
            v = [float(x) for x in chunk.replace(",", " ").split()]
            if len(v) != 6:
                raise SystemExit(f"anchor needs 6 numbers (x y z r p y), got: {chunk!r}")
            anchors.append((tuple(v[:3]), tuple(v[3:])))
    else:
        anchors = default_anchors(args.n, args.splay_x, args.splay_y)
    if len(anchors) != args.n:
        raise SystemExit(f"{len(anchors)} anchors for {args.n} MAVs")

    # strip every Falcon and rope element; keep payload and anything else
    for el in list(root):
        name = el.get("name", "") or ""
        if el.tag in ("link", "joint") and (FALCON_RE.match(name) or ROPE_RE.match(name)):
            root.remove(el)

    # the joint that attaches a cable to the payload, so we can set its anchor
    anchor_joint_re = re.compile(rf"^rope_{src}_sphere_joint_0_joint_x$")

    for i in range(1, args.n + 1):
        for el in falcon_tpl:
            root.append(renumber(el, src, i))
        (ax, ay, az), (ar, ap_, aya) = anchors[i - 1]
        for el in rope_tpl:
            new = renumber(el, src, i)
            if el.tag == "joint" and anchor_joint_re.match(el.get("name", "")):
                org = new.find("origin")
                if org is None:
                    org = ET.SubElement(new, "origin")
                org.set("xyz", f"{ax:.6g} {ay:.6g} {az:.6g}")
                org.set("rpy", f"{ar:.6g} {ap_:.6g} {aya:.6g}")
            root.append(new)

    if args.robot_name:
        root.set("name", args.robot_name)

    tree.write(args.outfile, xml_declaration=True, encoding="utf-8")

    links = sum(1 for e in root if e.tag == "link")
    joints = sum(1 for e in root if e.tag == "joint")
    print(f"wrote {args.outfile}: {args.n} MAVs, {links} links, {joints} joints")
    print("anchors (x y z / roll pitch yaw):")
    for i, ((x, y, z), (r, p, yw)) in enumerate(anchors, 1):
        print(f"  cable {i}: {x:+.3f} {y:+.3f} {z:+.3f}  /  {r:+.2f} {p:+.2f} {yw:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
