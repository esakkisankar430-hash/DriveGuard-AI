import cv2
import numpy as np


class HeadPoseEstimator:

    def __init__(self):
        pass

    # =========================================================
    # HEAD POSE USING MEDIAPIPE TRANSFORMATION MATRIX
    # =========================================================

    def get_head_pose(self, matrix):

        matrix = np.asarray(matrix, dtype=np.float64)

        # Make sure we have a 4x4 matrix
        if matrix.size != 16:
            return 0.0, 0.0, 0.0

        matrix = matrix.reshape(4, 4)

        rotation_matrix = matrix[:3, :3]

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            pitch = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            pitch = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = 0

        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return pitch, yaw, roll

    # =========================================================
    # CLASSIFY DIRECTION
    # =========================================================

    def classify_direction(self, pitch, yaw):

        # These signs were adjusted according to
        # your earlier webcam test.

        if pitch < -15:
            return "UP"

        elif pitch > 20:
            return "DOWN"

        elif yaw > 15:
            return "LEFT"

        elif yaw < -15:
            return "RIGHT"

        else:
            return "FORWARD"

    # =========================================================
    # ESTIMATE
    # =========================================================

    def estimate(self, face_result):

        # No face
        if face_result is None:

            return {
                "direction": "UNKNOWN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            }

        # No face landmarks
        if not face_result.face_landmarks:

            return {
                "direction": "UNKNOWN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            }

        # No transformation matrix
        if not face_result.facial_transformation_matrixes:

            return {
                "direction": "UNKNOWN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            }

        # Get first face transformation matrix
        matrix = face_result.facial_transformation_matrixes[0]

        pitch, yaw, roll = self.get_head_pose(
            matrix
        )

        direction = self.classify_direction(
            pitch,
            yaw
        )

        return {
            "direction": direction,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }


# =============================================================
# TEST PROGRAM
# =============================================================

def main():

    from face_detector import FaceDetector

    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()

    head_pose = HeadPoseEstimator()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ---------------------------------------------
        # Face detection
        # ---------------------------------------------

        face_result = face_detector.detect(
            frame
        )

        # ---------------------------------------------
        # Draw face landmarks
        # ---------------------------------------------

        frame = face_detector.draw_landmarks(
            frame,
            face_result
        )

        # ---------------------------------------------
        # Head pose
        # ---------------------------------------------

        pose = head_pose.estimate(
            face_result
        )

        direction = pose["direction"]

        pitch = pose["pitch"]

        yaw = pose["yaw"]

        roll = pose["roll"]

        # ---------------------------------------------
        # Direction
        # ---------------------------------------------

        cv2.putText(
            frame,
            f"Direction: {direction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ---------------------------------------------
        # Pitch
        # ---------------------------------------------

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------
        # Yaw
        # ---------------------------------------------

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------
        # Roll
        # ---------------------------------------------

        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------
        # Quit instruction
        # ---------------------------------------------

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
            "DriveGuard AI - Head Pose",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    face_detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
