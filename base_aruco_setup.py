"""Base setup: average the camera -> tag-1 transform over 100 frames.

Opens the ZED, collects 100 valid frames (tag id 1 visible), saves the
averaged 4x4 camera-to-tag-1 matrix to file, and exits. Units: cm.
"""

from pathlib import Path

import cv2
import numpy as np

from aruco_pose import ZedArucoPose

ORIGIN_ID = 1
N_FRAMES = 100
OUTPUT_FILE = Path(__file__).resolve().parent / "T_camera_to_tag1.npy"


def average_transforms(R_list, t_list):
    """Average rotations (SVD re-orthonormalized) and translations."""
    R_avg = sum(R_list) / len(R_list)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:  # guard against reflection
        Vt[-1, :] *= -1
        R_avg = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R_avg
    T[:3, 3] = sum(t_list) / len(t_list)
    return T


def main():
    est = ZedArucoPose(marker_length_cm=5.0)  # set your tag size in cm
    R_list, t_list = [], []
    with est:
        print("ZED opened. Hold tag 1 still in view. Press Q/ESC to abort.")
        while len(R_list) < N_FRAMES:
            frame = est.grab()
            if frame is None:
                continue
            poses = est.detect_and_estimate(frame)
            origin = next((p for p in poses if p["id"] == ORIGIN_ID), None)
            if origin is not None:
                R, _ = cv2.Rodrigues(origin["rvec"])
                R_list.append(R.T)  # tag1 orientation seen from camera
                t_list.append((-R.T @ origin["tvec"]).reshape(3))
                # # Mark the id-1 tag on screen.
                # pts = origin["corners"].astype(int).reshape(-1, 1, 2)
                # cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                # cv2.drawFrameAxes(frame, est.CAMERA_MATRIX, est.DIST_COEFFS,
                #                   origin["rvec"], origin["tvec"],
                #                   est.marker_length_cm / 2.0)
                # x, y = pts[0][0]
                # cv2.putText(frame, "tag 1",
                #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{len(R_list)}/{N_FRAMES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Base ArUco Setup", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Aborted.")
                break
    cv2.destroyAllWindows()
    if not R_list:
        print("No valid frames collected.")
        return
    T = average_transforms(R_list, t_list)
    np.save(OUTPUT_FILE, T)
    np.set_printoptions(precision=3, suppress=True)
    print(f"Collected {len(R_list)}/{N_FRAMES} valid frames.")
    print("T_camera_to_tag1 (cm):")
    print(T)
    print("position (x, y, z) cm:", T[:3, 3].tolist())
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
