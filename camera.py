import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open the camera.")

    def read(self):
        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Could not read frame from camera.")

        return frame

    def release(self):
        self.camera.release()


def main():
    camera = Camera()

    print("DriveGuard AI Camera Test")
    print("Press Q to quit.")

    while True:
        frame = camera.read()

        cv2.imshow("DriveGuard AI - Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
