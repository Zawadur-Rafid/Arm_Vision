"""Tag-1 relative coordinates. Origin = center of ArUco tag id 1.

Reuses ZedArucoPose from aruco_pose.py unchanged; only the displayed
(x, y, z) values are expressed in tag-1's frame instead of the camera's.
Units: cm.
"""

import cv2
import numpy as np

from aruco_pose import ZedArucoPose

ORIGIN_ID = 1


def relative_to_origin(poses, origin_id=ORIGIN_ID):
    """Express each tag's center in the origin tag's frame.

    Returns (rel_list, found). rel_list items: {id, t_rel, corners}.
    Origin tag itself maps to (0, 0, 0).
    """
    origin = next((p for p in poses if p["id"] == origin_id), None)
    if origin is None:
        return [], False
    R1, _ = cv2.Rodrigues(origin["rvec"])
    t1 = origin["tvec"]
    rel = []
    for p in poses:
        t_rel = R1.T @ (p["tvec"] - t1)
        rel.append({"id": p["id"], "t_rel": t_rel.reshape(3),
                    "corners": p["corners"]})
    return rel, True


def draw_relative(frame, est, poses, rel):
    """Draw outlines + pose axes + tag-1-relative xyz (cm) on the frame."""
    for p in poses:
        pts = p["corners"].astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.drawFrameAxes(frame, est.CAMERA_MATRIX, est.DIST_COEFFS,
                          p["rvec"], p["tvec"],
                          est.marker_length_cm / 2.0)
    lookup = {r["id"]: r["t_rel"] for r in rel}
    for p in poses:
        x, y = p["corners"].astype(int).reshape(4, 2)[0]
        cv2.putText(frame, f"id={p['id']}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        if p["id"] in lookup:
            t = lookup[p["id"]]
            cv2.putText(frame, f"({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) cm",
                        (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    if rel:
        cv2.putText(frame, "origin = tag 1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "tag 1 not visible", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def main():
    est = ZedArucoPose(marker_length_cm=5.0)  # set your tag size in cm
    with est:
        print("ZED opened. Press Q or ESC to quit.")
        while True:
            frame = est.grab()
            if frame is None:
                continue
            poses = est.detect_and_estimate(frame)
            rel, _ = relative_to_origin(poses)
            draw_relative(frame, est, poses, rel)
            cv2.imshow("ZED ArUco (origin = tag 1)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
