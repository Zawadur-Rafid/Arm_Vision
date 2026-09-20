"""Find reachable range along one Cartesian axis, holding the other two fixed.

Usage:
    python3 find_range.py --axis x --y 0 --z 9.5
    python3 find_range.py --axis y --x 20 --z 9.5
    python3 find_range.py --axis z --x 20 --y 0 --servo-aware --csv scan.csv
"""
import argparse
import csv
import math
import sys

from kinematics import EezybotKinematics

YAW_LIMIT_DEG = 45.0  # MK2 waist +/-45deg (URDF joint_1, 2:1 gear)


def build_point(axis, v, fx, fy, fz):
    if axis == "x":
        return v, fy, fz
    if axis == "y":
        return fx, v, fz
    return fx, fy, v


def point_ok(ik, x, y, z, servo_aware):
    """Returns (ok, reason). IK-only by default; servo-aware also rejects clamping."""
    ok, msg = ik.is_reachable(x, y, z)
    if not ok:
        return False, msg
    if not servo_aware:
        return True, "OK"
    try:
        t1, t2, t3 = ik.solve_ik(x, y, z)
    except ValueError as e:
        return False, str(e)
    t1d, t2d, t3d = math.degrees(t1), math.degrees(t2), math.degrees(t3)
    raw_m1 = ik.base_offset - ik.base_ratio * t1d
    raw_m2 = t2d + ik.shoulder_offset
    raw_m3 = ik.elbow_offset - t3d + t2d
    if abs(t1d) > YAW_LIMIT_DEG + 1e-9:
        return False, f"yaw {t1d:.1f}deg exceeds +-{YAW_LIMIT_DEG:.0f}deg"
    for name, raw in (("base", raw_m1), ("shoulder", raw_m2), ("elbow", raw_m3)):
        if not 0.0 <= raw <= 180.0:
            return False, f"{name} servo {raw:.1f} out of [0,180]"
    return True, "OK"


def refine_edge(ik, v_false, v_true, axis, fx, fy, fz, servo_aware, tol=0.1):
    """Binary-search the False->True boundary to within tol."""
    lo, hi = v_false, v_true
    if hi < lo:
        lo, hi = hi, lo
        # keep track: hi is True side
    # ensure hi is the True side
    x, y, z = build_point(axis, hi, fx, fy, fz)
    ok, _ = point_ok(ik, x, y, z, servo_aware)
    if not ok:
        lo, hi = hi, lo  # swap so hi is True
    for _ in range(20):
        if abs(hi - lo) < tol:
            break
        mid = 0.5 * (lo + hi)
        x, y, z = build_point(axis, mid, fx, fy, fz)
        ok, _ = point_ok(ik, x, y, z, servo_aware)
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser(description="Sweep one axis for reachable range.")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x")
    ap.add_argument("--x", type=float, default=20.0, help="fixed X (used when axis is y/z)")
    ap.add_argument("--y", type=float, default=0.0, help="fixed Y (used when axis is x/z)")
    ap.add_argument("--z", type=float, default=9.5, help="fixed Z in cm (used when axis is x/y)")
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.5, help="coarse scan step in cm")
    ap.add_argument("--tol", type=float, default=0.1, help="boundary refine tolerance in cm")
    ap.add_argument("--servo-aware", action="store_true",
                    help="reject points needing servo clamping / yaw > +-45deg")
    ap.add_argument("--csv", type=str, default=None, help="optional output CSV path")
    args = ap.parse_args()

    axis = args.axis
    fx, fy, fz = args.x, args.y, args.z
    lo, hi = args.lo, args.hi
    if lo is None or hi is None:
        defaults = {"x": (0.0, 40.0), "y": (-40.0, 40.0), "z": (0.0, 40.0)}
        dlo, dhi = defaults[axis]
        lo = dlo if lo is None else lo
        hi = dhi if hi is None else hi
    if hi <= lo or args.step <= 0:
        sys.exit("lo must be < hi and step > 0")

    ik = EezybotKinematics()
    fixed_desc = {"x": f"y={fy}, z={fz}", "y": f"x={fx}, z={fz}", "z": f"x={fx}, y={fy}"}[axis]
    print(f"Sweep {axis} in [{lo:.1f}, {hi:.1f}] step {args.step} @ {fixed_desc} "
          f"({'servo-aware' if args.servo_aware else 'IK-only'})")

    # 1. Coarse scan
    n = int(round((hi - lo) / args.step)) + 1
    values, oks, reasons = [], [], []
    for i in range(n):
        v = lo + i * args.step
        if v > hi:
            v = hi
        x, y, z = build_point(axis, v, fx, fy, fz)
        ok, reason = point_ok(ik, x, y, z, args.servo_aware)
        values.append(v)
        oks.append(ok)
        reasons.append(reason)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sweep_value", "x", "y", "z", "reachable", "reason"])
            for v, ok, reason in zip(values, oks, reasons):
                x, y, z = build_point(axis, v, fx, fy, fz)
                w.writerow([f"{v:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", int(ok), reason])
        print(f"Wrote coarse scan to {args.csv}")

    # 2. Contiguous intervals (handles base-cylinder / inner-void gaps)
    intervals_idx = []
    start = None
    for i, ok in enumerate(oks):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            intervals_idx.append((start, i - 1))
            start = None
    if start is not None:
        intervals_idx.append((start, len(oks) - 1))

    if not intervals_idx:
        print("No reachable points in range.")
        print(f"Example reason @ mid {(lo+hi)/2:.1f}: {reasons[len(reasons)//2]}")
        return

    # 3. Refine boundaries + validate mid with FK
    print(f"Found {len(intervals_idx)} reachable interval(s):")
    for k, (a, b) in enumerate(intervals_idx):
        v0, v1 = values[a], values[b]
        if a > 0:
            v0 = refine_edge(ik, values[a - 1], values[a], axis, fx, fy, fz,
                             args.servo_aware, tol=args.tol)
        if b < len(values) - 1:
            v1 = refine_edge(ik, values[b + 1], values[b], axis, fx, fy, fz,
                             args.servo_aware, tol=args.tol)
        mid = 0.5 * (v0 + v1)
        x, y, z = build_point(axis, mid, fx, fy, fz)
        try:
            t = ik.solve_ik(x, y, z)
            f = ik.solve_fk(*t)
            err = math.dist((x, y, z), f)
            m = ik.joint_angles_to_servo_degrees(*t)
            print(f"  [{k+1}] {axis} in [{v0:.1f}, {v1:.1f}] cm "
                  f"(width {v1-v0:.1f})  mid FK err {err:.3f}cm  servo@{mid:.1f}={m}")
        except ValueError as e:
            print(f"  [{k+1}] {axis} in [{v0:.1f}, {v1:.1f}] cm  (re-solve failed: {e})")


if __name__ == "__main__":
    main()
