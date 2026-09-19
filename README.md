# arm

ArUco marker pose tracking with a Stereolabs ZED camera. The scripts detect 4x4 ArUco markers from the ZED left camera, estimate their 3D pose in centimetres, and optionally express positions relative to marker 1 or a saved base frame.

## Requirements

- Python 3.10 or newer
- A supported Stereolabs ZED camera
- ZED SDK with its Python API (`pyzed.sl`) installed
- OpenCV with the ArUco module and NumPy

Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

Install the ZED SDK separately from the [Stereolabs downloads page](https://www.stereolabs.com/developers/release/), then make sure `import pyzed.sl` works in the Python environment used to run these scripts.

## Scripts

- `zedcam.py` displays the ZED left-camera stream.
- `aruco_pose.py` detects markers and displays each marker's camera-relative pose.
- `aruco_relative.py` expresses marker positions relative to visible marker ID 1.
- `base_aruco_setup.py` averages 100 observations of marker ID 1 and saves `T_camera_to_tag1.npy`.
- `aruco_base_relative.py` loads that matrix and displays marker positions in the saved base frame.

All examples currently use a 5 cm marker. Change `marker_length_cm` in the script entry points when using a different printed marker size.

## Usage

Run from the project directory with the ZED camera connected:

```bash
python zedcam.py
python aruco_pose.py
python aruco_relative.py
```

To use a fixed base frame, first hold marker 1 still and run:

```bash
python base_aruco_setup.py
python aruco_base_relative.py
```

Press `q` or `Esc` to stop any camera window.

## Coordinate conventions

- Distances are reported in centimetres.
- Marker IDs use OpenCV's predefined `DICT_4X4_50` dictionary.
- The camera calibration constants in `aruco_pose.py` are for the ZED HD720 left camera.
- `T_camera_to_tag1.npy` is generated locally by the base setup script and is intentionally ignored by Git.

## Hardware notes

The camera must remain fixed relative to the base after `base_aruco_setup.py` creates the transform. Use a correctly printed ArUco marker with the physical size configured in the script for meaningful pose estimates.
