import cv2
import time
import math


class DrowsinessDetector:

    # MediaPipe Face Landmarker eye points
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Prototype thresholds
    EAR_THRESHOLD = 0.21
    CLOSED_TIME_THRESHOLD = 1.5

    def __init__(self):
        self.eyes_closed_start = None

    def calculate_ear(self, landmarks, eye_points):

        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]

        # Vertical eye distances
        vertical_1 = math.sqrt(
            (p2.x - p6.x) ** 2 +
            (p2.y - p6.y) ** 2
        )

        vertical_2 = math.sqrt(
            (p3.x - p5.x) ** 2 +
            (p3.y - p5.y) ** 2
        )

        # Horizontal eye distance
        horizontal = math.sqrt(
            (p1.x - p4.x) ** 2 +
            (p1.y - p4.y) ** 2
        )

        if horizontal == 0:
            return 0.0

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        return ear

    def detect(self, face_result):

        result = {
            "drowsy": False,
            "eyes_closed": False,
            "ear": 0.0,
            "closed_duration": 0.0
        }

        # No face detected
        if face_result is None:
            self.eyes_closed_start = None
            return result

        if not face_result.face_landmarks:
            self.eyes_closed_start = None
            return result

        # Use first detected face
        landmarks = face_result.face_landmarks[0]

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(
            landmarks,
            self.LEFT_EYE
        )

        right_ear = self.calculate_ear(
            landmarks,
            self.RIGHT_EYE
        )

        # Average EAR
        ear = (left_ear + right_ear) / 2.0

        result["ear"] = ear

        # Check whether eyes are closed
        if ear < self.EAR_THRESHOLD:

            result["eyes_closed"] = True

            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()

            closed_duration = (
                time.time() - self.eyes_closed_start
            )

            result["closed_duration"] = closed_duration

            # Drowsiness condition
            if closed_duration >= self.CLOSED_TIME_THRESHOLD:
                result["drowsy"] = True

        else:

            # Eyes opened
            self.eyes_closed_start = None
            result["eyes_closed"] = False
            result["closed_duration"] = 0.0

        return result


def main():

    from face_detector import FaceDetector

    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    drowsiness_detector = DrowsinessDetector()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Face detection
        face_result = face_detector.detect(frame)

        # Draw facial landmarks
        frame = face_detector.draw_landmarks(
            frame,
            face_result
        )

        # Drowsiness detection
        drowsiness = drowsiness_detector.detect(
            face_result
        )

        ear = drowsiness["ear"]
        eyes_closed = drowsiness["eyes_closed"]
        drowsy = drowsiness["drowsy"]
        closed_duration = drowsiness["closed_duration"]

        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Display eye status
        if eyes_closed:

            cv2.putText(
                frame,
                "EYES CLOSED",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Closed: {closed_duration:.1f}s",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "EYES OPEN",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Drowsiness warning
        if drowsy:

            cv2.putText(
                frame,
                "DROWSINESS DETECTED!",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "ALERT",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        cv2.imshow(
            "DriveGuard AI - Drowsiness Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    face_detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
