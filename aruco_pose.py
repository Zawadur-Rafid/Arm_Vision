"""ZED left-camera ArUco detection + pose. Simple version."""

import cv2
import numpy as np
import pyzed.sl as sl


class ZedArucoPose:
    # LEFT_CAM_HD calibration from SN29966020.conf (720p)
    CAMERA_MATRIX = np.array(
        [[528.545, 0.0, 637.99],
         [0.0, 528.305, 359.4695],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    DIST_COEFFS = np.array(
        [-0.0398125, 0.00835987, -0.00040137, 0.000400917, -0.00444386],
        dtype=np.float64,
    )

    def __init__(self, marker_length_cm,
                  marker_sizes_cm=None,
                  dictionary=cv2.aruco.DICT_4X4_50):
        """Create estimator.

        Args:
            marker_length_cm: default physical tag size in cm, used for
                any id not listed in marker_sizes_cm. Kept for
                backwards compatibility.
            marker_sizes_cm: optional dict {marker_id: size_cm} for
                mixed-size setups, e.g. {1: 5.0} with default 3.0 when
                the base tag is 5 cm and all other tags are 3 cm.
        """
        self.default_marker_length_cm = float(marker_length_cm)
        # Backwards-compat alias: old code uses est.marker_length_cm.
        self.marker_length_cm = self.default_marker_length_cm
        self.marker_sizes_cm = (
            {int(k): float(v) for k, v in marker_sizes_cm.items()}
            if marker_sizes_cm else {}
        )
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary),
            cv2.aruco.DetectorParameters(),
        )
        s = self.default_marker_length_cm / 2.0
        self._obj_points = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
            dtype=np.float64,
        )
        self.zed = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = sl.RESOLUTION.HD720
        params.camera_fps = 30
        params.depth_mode = sl.DEPTH_MODE.NONE
        self._params = params
        self._runtime = sl.RuntimeParameters()
        self._mat = sl.Mat()
        self._opened = False

    def marker_length_for(self, marker_id):
        """Return physical size (cm) for a tag id (default if unknown)."""
        return self.marker_sizes_cm.get(int(marker_id),
                                        self.default_marker_length_cm)

    def _obj_points_for(self, marker_id):
        s = self.marker_length_for(marker_id) / 2.0
        return np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
            dtype=np.float64,
        )

    def open(self):
        status = self.zed.open(self._params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")
        self._opened = True
        return self

    def close(self):
        if self._opened:
            self.zed.close()
            self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def grab(self):
        """Grab one left image. Returns BGR frame or None."""
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_image(self._mat, sl.VIEW.LEFT)
        return cv2.cvtColor(self._mat.get_data(), cv2.COLOR_BGRA2BGR)

    def detect_and_estimate(self, frame):
        """Returns list of {id, rvec, tvec, corners} (tvec in cm)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = []
        if ids is None:
            return poses
        for corner, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_points_for(marker_id),
                np.asarray(corner, dtype=np.float64).reshape(4, 2),
                self.CAMERA_MATRIX,
                self.DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                poses.append({"id": marker_id,
                              "rvec": rvec.reshape(3),
                              "tvec": tvec.reshape(3),
                              "corners": np.asarray(corner).reshape(4, 2)})
        return poses

    def draw_poses(self, frame, poses):
        """Draw marker outlines + pose axes + id + position (cm)."""
        for p in poses:
            pts = p["corners"].astype(int).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, self.CAMERA_MATRIX, self.DIST_COEFFS,
                              p["rvec"], p["tvec"],
                              self.marker_length_for(p["id"]) / 2.0)
            x, y = pts[0][0]
            t = p["tvec"]
            cv2.putText(frame, f"id={p['id']}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) cm",
                        (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
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
            est.draw_poses(frame, poses)
            cv2.imshow("ZED ArUco", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
