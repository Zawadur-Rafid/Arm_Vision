import pyzed.sl as sl
import cv2


def main():
    # Create ZED camera
    zed = sl.Camera()

    # Camera initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # Open the camera
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        return

    print("ZED camera opened successfully.")

    # Create an image container
    image = sl.Mat()

    runtime_params = sl.RuntimeParameters()

    try:
        while True:
            # Grab a new frame
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left camera image
                zed.retrieve_image(image, sl.VIEW.LEFT)

                # Convert ZED image to NumPy/OpenCV format
                frame = image.get_data()

                # ZED returns BGRA; OpenCV uses BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Display frame
                cv2.imshow("ZED Camera", frame)

            # Press Q or ESC to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        # Clean up
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
