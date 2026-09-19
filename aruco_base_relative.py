"""Show tag positions relative to the saved base (tag 1) frame.

Loads T_camera_to_tag1.npy written by base_aruco_setup.py, opens the ZED,
and draws each visible tag's id + its (x, y, z) in the base frame (cm).
The camera must stay fixed relative to where it was during base setup.
Tag 1 itself does not need to stay visible.
"""

from pathlib import Path

import cv2
import numpy as np

from aruco_pose import ZedArucoPose

MATRIX_FILE = Path(__file__).resolve().parent / "T_camera_to_tag1.npy"

# Base tag (id 1) printed at 5 cm, all other tags printed at 3 cm.
# base_aruco_setup.py estimates T_camera_to_tag1 with the 5 cm size, so we
# must keep that size for id 1 here and use 3 cm for everything else.
# Using 5 cm for a 3 cm tag would scale its distance by ~5/3.
BASE_ID = 1
BASE_MARKER_LENGTH_CM = 5.0
OTHER_MARKER_LENGTH_CM = 3.0
# Number of valid frames to average for capture_best_target().
DEFAULT_N_FRAMES = 10


def load_base_matrix(path=MATRIX_FILE):
    T = np.load(str(path))
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got {T.shape}")
    return T


def to_base_frame(tvec_cam, T):
    """Map a tag center from camera frame to base frame (cm)."""
    return (T[:3, :3] @ np.asarray(tvec_cam).reshape(3) + T[:3, 3])


def draw_base_relative(frame, est, poses, T):
    """Draw outlines + id + base-frame xyz (cm) on the frame."""
    for p in poses:
        pts = p["corners"].astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.drawFrameAxes(frame, est.CAMERA_MATRIX, est.DIST_COEFFS,
                          p["rvec"], p["tvec"],
                          est.marker_length_for(p["id"]) / 2.0)
        x, y = pts[0][0]
        b = to_base_frame(p["tvec"], T)
        cv2.putText(frame, f"id={p['id']}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}) cm",
                    (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return frame


def select_best_candidate(candidates):
    """Select one candidate: smallest x; x-tie -> smallest distance.

    Args:
        candidates: list of dicts with keys id, x, y, z (base frame, cm).

    Returns:
        The winning dict, or None if the list is empty.
        Comparison uses raw floats (no rounding). Distance is the 3D
        distance from the base-frame origin: sqrt(x^2 + y^2 + z^2).
    """
    if not candidates:
        return None
    best, best_d2 = None, None
    for c in candidates:
        d2 = float(c["x"]) ** 2 + float(c["y"]) ** 2 + float(c["z"]) ** 2
        if (best is None or float(c["x"]) < float(best["x"])
                or (float(c["x"]) == float(best["x"]) and d2 < best_d2)):
            best, best_d2 = c, d2
    return best


def poses_to_base_candidates(poses, T, exclude_ids=(BASE_ID,)):
    """Convert raw poses to base-frame candidates, excluding base id(s).

    Returns:
        List of {id, x, y, z} dicts (floats, cm).
    """
    excluded = set(int(i) for i in exclude_ids)
    out = []
    for p in poses:
        if int(p["id"]) in excluded:
            continue
        b = to_base_frame(p["tvec"], T)
        out.append({"id": int(p["id"]), "x": float(b[0]),
                    "y": float(b[1]), "z": float(b[2])})
    return out


def capture_best_target(T=None, warmup_frames=5, n_frames=DEFAULT_N_FRAMES,
                        max_grab_attempts=90):
    """Open the ZED, average N frames, pick one tag, then close the camera.

    Opens the camera, discards ``warmup_frames`` grabs (auto-exposure
    often needs a few frames), then collects up to ``n_frames`` valid
    frames (frames yielding at least one non-base tag). Each visible
    non-base tag's base-frame (x, y, z) is averaged across the frames it
    appeared in, then the winner is selected: smallest averaged x,
    tie -> smallest 3D distance from the base origin. Tag id 1 is always
    excluded.

    Args:
        T: optional pre-loaded 4x4 base matrix (loads from file if None).
        warmup_frames: grabs to discard right after opening the camera.
        n_frames: number of valid frames to average (default 10).
        max_grab_attempts: max frames to inspect (after warmup) before
            giving up. ``None`` returns from any of these paths mean
            "no aruco code available".

    Returns:
        Dict {id, x, y, z, n_obs, frames_used} for the winner, or None
        if no tag found. x/y/z are per-tag averages (cm, floats);
        n_obs is how many of the averaged frames contained the winner;
        frames_used is how many valid frames were averaged in total.
        The camera is always closed before returning (context manager).
    """
    if T is None:
        T = load_base_matrix()
    n_frames = max(1, int(n_frames))
    est = ZedArucoPose(
        marker_length_cm=OTHER_MARKER_LENGTH_CM,
        marker_sizes_cm={BASE_ID: BASE_MARKER_LENGTH_CM},
    )
    # id -> [sum_x, sum_y, sum_z, count]
    sums = {}
    frames_used = 0
    with est:  # camera ON here, OFF on block exit
        for _ in range(max(0, int(warmup_frames))):
            est.grab()  # discard; ignore failures during warm-up
        for _ in range(max(n_frames, int(max_grab_attempts))):
            if frames_used >= n_frames:
                break
            frame = est.grab()
            if frame is None:
                continue
            poses = est.detect_and_estimate(frame)
            candidates = poses_to_base_candidates(poses, T)
            if not candidates:
                continue  # nothing usable in this frame; try next one
            frames_used += 1
            for c in candidates:
                s = sums.setdefault(int(c["id"]), [0.0, 0.0, 0.0, 0])
                s[0] += float(c["x"])
                s[1] += float(c["y"])
                s[2] += float(c["z"])
                s[3] += 1
    if not sums or frames_used == 0:
        return None
    averaged = [{"id": tid,
                 "x": s[0] / s[3], "y": s[1] / s[3], "z": s[2] / s[3],
                 "n_obs": s[3]}
                for tid, s in sums.items()]
    best = select_best_candidate(averaged)
    best["frames_used"] = frames_used
    return best


def main():
    T = load_base_matrix()
    np.set_printoptions(precision=3, suppress=True)
    print("Loaded base matrix:")
    print(T)
    print(f"Base tag id {BASE_ID}: {BASE_MARKER_LENGTH_CM} cm, "
          f"other tags: {OTHER_MARKER_LENGTH_CM} cm")
    est = ZedArucoPose(
        marker_length_cm=OTHER_MARKER_LENGTH_CM,
        marker_sizes_cm={BASE_ID: BASE_MARKER_LENGTH_CM},
    )
    with est:
        print("ZED opened. Press Q or ESC to quit.")
        while True:
            frame = est.grab()
            if frame is None:
                continue
            poses = est.detect_and_estimate(frame)
            draw_base_relative(frame, est, poses, T)
            cv2.imshow("ZED ArUco (base frame)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
