"""Camera (ArUco) XY -> servo angles. Test/demo mapper, no hardware required.

Default run (``python servo_angle_mapper.py``) does a single shot:
  1. Open the ZED via aruco_base_relative.capture_best_target(), average
     10 frames, keep every tag except id 1, select one (smallest averaged
     x; x-tie -> smallest 3D distance from the base origin), return its
     id + averaged x,y,z, then CLOSE the camera.
  2. With the camera off, map the selected X,Y through the calibration
     table to (lower, base, upper) servo angles and print them.

Pipeline details (per selected tag):
   1. Selection uses raw floats, BEFORE truncation (done inside
      aruco_base_relative for the camera path).
   2. Truncate selected X,Y toward zero (math.trunc).
   3. Map X by floor: largest calibration X <= X_processed.
   4. Map Y by nearest neighbour among rows with X == X_map.
      Tie -> smaller Y (deterministic).
   5. Retrieve (lower, base, upper) in degrees directly from the table.
      No claw, no Z, no interpolation, no regression.

Units: X,Y,Z in cm; servo angles in degrees.
Use ``--demo`` for the offline sample data, ``--live`` for the old
continuous-preview mode (camera stays open).
"""

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Calibration dataset (SL, X, Y, Lower, Base, Upper). Angles in degrees.
# Intentionally NO claw column, NO Z column.
# ---------------------------------------------------------------------------
# (sl, x, y, lower, base, upper)
CALIBRATION_ROWS = [
    (1, 34, 2, 160, 70, 50),
    (2, 34, -1, 150, 80, 55),
    (3, 34, -5, 155, 93, 50),
    (4, 33, -8, 160, 105, 55),
    (5, 31, 2, 140, 70, 40),
    (6, 30, -1, 140, 80, 40),
    (7, 30, -4, 140, 93, 40),
    (8, 30, -8, 140, 107, 40),
    (9, 27, 3, 130, 65, 30),
    (10, 27, 0, 130, 80, 30),
    (11, 27, -4, 130, 93, 30),
    (12, 26, -7, 130, 107, 30),
    (13, 25, 3, 125, 65, 30),
    (14, 24, 0, 125, 80, 25),
    (15, 24, -4, 125, 93, 20),
    (16, 24, -7, 127, 112, 20),
    (17, 34, 0, 160, 75, 50),
    (18, 34, -3, 155, 85, 50),
    (19, 33, -6, 160, 100, 55),
    (20, 31, 1, 140, 75, 40),
    (21, 30, -2, 140, 85, 40),
    (22, 30, -6, 140, 97, 40),
    (23, 27, 1, 130, 70, 30),
    (24, 27, -2, 135, 85, 25),
    (25, 26, -6, 135, 100, 30),
    (26, 25, 2, 125, 70, 25),
    (27, 24, -2, 125, 85, 25),
    (28, 24, -5, 127, 100, 20),
    (29, 32, 2, 160, 70, 50),
    (30, 32, -1, 150, 83, 45),
    (31, 32, -5, 150, 93, 45),
    (32, 32, -8, 150, 105, 50),
    (33, 29, 3, 140, 67, 40),
    (34, 29, -1, 140, 80, 35),
    (35, 28, -4, 135, 93, 30),
    (36, 28, -7, 135, 110, 35),
    (37, 26, 3, 125, 67, 30),
    (38, 25, 0, 130, 80, 25),
    (39, 25, -4, 130, 93, 20),
    (40, 24, -7, 130, 110, 20),
    (41, 32, 1, 155, 75, 50),
    (42, 32, -3, 140, 90, 45),
    (43, 32, -6, 150, 100, 40),
    (44, 29, 1, 135, 73, 35),
    (45, 28, -2, 130, 90, 40),
    (46, 28, -6, 135, 100, 30),
    (47, 26, 1, 130, 73, 30),
    (48, 25, -2, 125, 90, 20),
    (49, 25, -5, 130, 100, 20),
]

# Precomputed lookup structures.
SORTED_XS = sorted({r[1] for r in CALIBRATION_ROWS})
# (x, y) -> first row with that key (lowest SL wins on duplicates,
# e.g. (23, -6) appears as SL16 and SL40; SL16 is kept).
ROW_BY_XY = {}
for _sl, _x, _y, _lo, _ba, _up in sorted(CALIBRATION_ROWS, key=lambda r: r[0]):
    ROW_BY_XY.setdefault((_x, _y), (_lo, _ba, _up))
# x -> sorted unique y values present for that x.
YS_BY_X = {x: sorted({r[2] for r in CALIBRATION_ROWS if r[1] == x})
           for x in SORTED_XS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _xy_of(det):
    """Extract (x, y) floats from a detection. Z is never read."""
    if isinstance(det, dict):
        # Accept x/y or X/Y keys; ignore everything else (id, z, Z, ...).
        x = det.get("x", det.get("X"))
        y = det.get("y", det.get("Y"))
        if x is None or y is None:
            raise ValueError(f"Detection dict needs x/y keys, got {det!r}")
        return float(x), float(y)
    # Tuple/list: (x, y) or (x, y, z) -- z element ignored if present.
    return float(det[0]), float(det[1])


def select_object(detections):
    """Select target object: min raw X, tie -> min distance from origin.

    Uses RAW camera values (no rounding). Returns (index, detection).
    Tie-break uses 3D distance sqrt(X^2+Y^2+Z^2) when Z is present,
    otherwise 2D distance sqrt(X^2+Y^2). Mapping itself still ignores Z.
    Raises ValueError on empty input.
    """
    if not detections:
        raise ValueError("No objects detected.")
    best_i, best_d2, best_x = None, None, None
    for i, det in enumerate(detections):
        x, y = _xy_of(det)
        z = None
        if isinstance(det, dict):
            z = det.get("z", det.get("Z"))
        elif isinstance(det, (list, tuple)) and len(det) >= 3:
            z = det[2]
        if z is None:
            d2 = x * x + y * y
        else:
            z = float(z)
            d2 = x * x + y * y + z * z  # squared 3D distance; monotonic vs D
        if (best_i is None or x < best_x
                or (x == best_x and d2 < best_d2)):
            best_i, best_d2, best_x = i, d2, x
    return best_i, detections[best_i]


def trunc_toward_zero(v):
    """Round toward zero: 2.8->2, -2.8->-2. Equivalent to math.trunc."""
    return math.trunc(float(v))


def map_x(x_processed):
    """Floor lookup: largest calibration X <= x_processed.

    Returns None if x_processed is below every calibration X
    (no valid floor exists; caller must not move the arm).
    """
    cands = [x for x in SORTED_XS if x <= x_processed]
    return max(cands) if cands else None


def map_y(x_map, y_processed):
    """Nearest Y among rows with X == x_map. Tie -> smaller Y.

    Deterministic: min by (|Yi - Yp|, Yi).
    """
    ys = YS_BY_X.get(x_map)
    if not ys:
        raise ValueError(f"No calibration rows with X={x_map}.")
    return min(ys, key=lambda yi: (abs(yi - y_processed), yi))


def get_angles(x_map, y_map):
    """Direct lookup of (lower, base, upper) in degrees. No interpolation."""
    try:
        return ROW_BY_XY[(x_map, y_map)]
    except KeyError:
        raise ValueError(f"No calibration row for ({x_map}, {y_map}).")


def pick_from_detections(detections):
    """Full pipeline. Returns dict with every intermediate value + angles.

    Result keys: selected_index, x_raw, y_raw, x_proc, y_proc,
    x_map, y_map, lower_deg, base_deg, upper_deg.
    """
    idx, det = select_object(detections)
    x_raw, y_raw = _xy_of(det)  # Z ignored: never read
    x_proc = trunc_toward_zero(x_raw)
    y_proc = trunc_toward_zero(y_raw)
    x_map = map_x(x_proc)
    if x_map is None:
        raise ValueError(
            f"X_processed={x_proc} below minimum calibration X={SORTED_XS[0]}; "
            "no floor X exists (refusing to move beyond table).")
    y_map = map_y(x_map, y_proc)
    lower, base, upper = get_angles(x_map, y_map)
    return {
        "selected_index": idx,
        "x_raw": x_raw, "y_raw": y_raw,
        "x_proc": x_proc, "y_proc": y_proc,
        "x_map": x_map, "y_map": y_map,
        "lower_deg": lower, "base_deg": base, "upper_deg": upper,
    }


# ---------------------------------------------------------------------------
# Test / demo (no hardware): prints the three servo angles for sample inputs.
# ---------------------------------------------------------------------------
def _demo():
    cases = [
        ("spec min-X",
         [{"x": 33.2, "y": 2.1}, {"x": 29.7, "y": -1.5}, {"x": 31.4, "y": -4.2}]),
        ("spec X-tie -> closest to origin",
         [{"x": 29.0, "y": -5.0}, {"x": 29.0, "y": -2.0}, {"x": 31.0, "y": 0.0}]),
        ("spec Y-tie: X_map=29 Y_proc=-3 -> -4 wins (smaller Y)",
         [{"x": 29.2, "y": -3.2}]),
        ("trunc demo: 29.7 -> 29; floor 31.8 -> 31",
         [{"x": 31.8, "y": -1.2}]),
        ("negative trunc: -2.8 -> -2",
         [{"x": 26.9, "y": -2.8}]),
    ]
    for name, dets in cases:
        try:
            r = pick_from_detections(dets)
            print(f"[{name}] sel#{r['selected_index']} "
                  f"raw=({r['x_raw']:.2f},{r['y_raw']:.2f}) "
                  f"proc=({r['x_proc']},{r['y_proc']}) "
                  f"map=({r['x_map']},{r['y_map']}) -> "
                  f"lower={r['lower_deg']} base={r['base_deg']} "
                  f"upper={r['upper_deg']}")
        except ValueError as e:
            print(f"[{name}] ERROR: {e}")
    print(f"Calibration X range {SORTED_XS[0]}..{SORTED_XS[-1]}; "
          f"duplicate (23,-6) resolves to first SL row {ROW_BY_XY[(23, -6)]}")


def _live():
    """Live mode: X,Y from aruco_base_relative (base frame, cm). Prints angles."""
    import cv2
    from aruco_base_relative import (
        BASE_ID, BASE_MARKER_LENGTH_CM, OTHER_MARKER_LENGTH_CM,
        load_base_matrix, to_base_frame,
    )
    from aruco_pose import ZedArucoPose

    T = load_base_matrix()
    est = ZedArucoPose(marker_length_cm=OTHER_MARKER_LENGTH_CM,
                       marker_sizes_cm={BASE_ID: BASE_MARKER_LENGTH_CM})
    with est:
        print("ZED opened. Press Q/ESC to quit.")
        while True:
            frame = est.grab()
            if frame is None:
                continue
            poses = est.detect_and_estimate(frame)
            # Exclude the base tag itself; only real objects are mappable.
            poses = [p for p in poses if int(p["id"]) != BASE_ID]
            if poses:
                # Base-frame XY (cm); Z dropped inside _xy_of / pipeline.
                dets = []
                for p in poses:
                    b = to_base_frame(p["tvec"], T)
                    dets.append({"id": p["id"], "x": float(b[0]),
                                 "y": float(b[1])})
                    pts = p["corners"].astype(int).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                try:
                    r = pick_from_detections(dets)
                    tid = dets[r["selected_index"]].get("id", "?")
                    cv2.putText(frame,
                                f"pick id={tid} "
                                f"({r['x_map']},{r['y_map']}) "
                                f"L={r['lower_deg']} B={r['base_deg']} "
                                f"U={r['upper_deg']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    print(f"pick id={tid} map=({r['x_map']},{r['y_map']}) "
                          f"-> lower={r['lower_deg']} "
                          f"base={r['base_deg']} upper={r['upper_deg']}")
                except ValueError as e:
                    cv2.putText(frame, str(e)[:60], (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "no tags", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Servo mapper (live)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


def _run_once():
    """Default mode: one-shot capture (cam on/off) -> angles.

    1. Camera ON: aruco_base_relative.capture_best_target() averages
       10 frames, drops id 1, selects smallest-x (tie: closest to
       origin), returns {id, x, y, z} averages, then closes the camera.
    2. Camera OFF: map that single detection to servo angles.
    Runs exactly once, then exits.
    """
    from aruco_base_relative import capture_best_target

    try:
        target = capture_best_target()  # cam on ... cam off inside
    except Exception as e:
        print(f"Capture failed: {e}")
        return
    if target is None:
        print("No aruco codes available.")
        return
    tid, x, y, z = target["id"], target["x"], target["y"], target["z"]
    print(f"Detected id={tid} x={x:.2f} y={y:.2f} z={z:.2f} cm "
          f"(camera now off).")
    try:
        # Single-element list: selection already done; this only
        # truncates + floor/nearest maps + table lookup.
        r = pick_from_detections([target])
    except ValueError as e:
        print(f"id={tid} ERROR: {e}")
        return
    print(f"id={tid} raw=({r['x_raw']:.2f},{r['y_raw']:.2f}) "
          f"proc=({r['x_proc']},{r['y_proc']}) "
          f"map=({r['x_map']},{r['y_map']}) -> "
          f"lower={r['lower_deg']} base={r['base_deg']} "
          f"upper={r['upper_deg']}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
    elif "--live" in sys.argv:
        _live()
    else:
        _run_once()
