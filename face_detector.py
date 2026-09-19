import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self):

        self.model_path = "models/face_landmarker.task"

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self.model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True
        )

        self.detector = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        self.timestamp = 0

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp += 1

        result = self.detector.detect_for_video(
            mp_image,
            self.timestamp
        )

        return result

    def draw_landmarks(self, frame, result):

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                for landmark in face_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        return frame

    def close(self):

        self.detector.close()


def main():

    cap = cv2.VideoCapture(0)

    detector = FaceDetector()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect(frame)

        frame = detector.draw_landmarks(
            frame,
            result
        )

        if result.face_landmarks:

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "DriveGuard AI - Face Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
