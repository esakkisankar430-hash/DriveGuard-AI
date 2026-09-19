import cv2

from face_detector import FaceDetector
from phone_detector import PhoneDetector
from head_pose import HeadPoseEstimator
from drowsiness import DrowsinessDetector
from risk_engine import RiskEngine


def main():

    # --------------------------------
    # INITIALIZE MODULES
    # --------------------------------

    face_detector = FaceDetector()
    phone_detector = PhoneDetector()
    head_pose = HeadPoseEstimator()
    drowsiness_detector = DrowsinessDetector()
    risk_engine = RiskEngine()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    print("DriveGuard AI - Step 9E")
    print("Risk Engine Integration")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        # ==================================
        # 1. FACE DETECTION
        # ==================================

        face_result = face_detector.detect(frame)

        frame = face_detector.draw_landmarks(
            frame,
            face_result
        )

        # ==================================
        # 2. PHONE DETECTION
        # ==================================

        phone_detected, phone_confidence, boxes = (
            phone_detector.detect(frame)
        )

        frame = phone_detector.draw_detections(
            frame,
            boxes
        )

        # ==================================
        # 3. HEAD POSE
        # ==================================

        pose = head_pose.estimate(
            face_result
        )

        head_direction = pose["direction"]

        # ==================================
        # 4. DROWSINESS
        # ==================================

        drowsiness = drowsiness_detector.detect(
            face_result
        )

        drowsy = drowsiness["drowsy"]

        # ==================================
        # 5. RISK ENGINE
        # ==================================

        risk = risk_engine.calculate_risk(
            phone_detected=phone_detected,
            head_direction=head_direction,
            drowsy=drowsy
        )

        score = risk["score"]
        level = risk["level"]
        events = risk["events"]

        # ==================================
        # DISPLAY INFORMATION
        # ==================================

        cv2.putText(
            frame,
            f"Head: {head_direction}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Phone: {'YES' if phone_detected else 'NO'}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drowsy: {'YES' if drowsy else 'NO'}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # RISK SCORE
        # --------------------------------

        cv2.putText(
            frame,
            f"RISK SCORE: {score}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"RISK LEVEL: {level}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if level == "LOW"
            else (0, 255, 255) if level == "MEDIUM"
            else (0, 0, 255),
            2
        )

        # ==================================
        # EVENTS
        # ==================================

        if events:

            y = 220

            for event in events:

                cv2.putText(
                    frame,
                    f"! {event}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                y += 30

        else:

            cv2.putText(
                frame,
                "No risk events",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ==================================
        # SHOW WINDOW
        # ==================================

        cv2.imshow(
            "DriveGuard AI - Risk Engine Integration",
            frame
        )

        # ==================================
        # QUIT
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==================================
    # CLEANUP
    # ==================================

    cap.release()

    face_detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
