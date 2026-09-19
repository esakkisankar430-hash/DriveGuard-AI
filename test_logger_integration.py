import cv2

from face_detector import FaceDetector
from phone_detector import PhoneDetector
from head_pose import HeadPoseEstimator
from drowsiness import DrowsinessDetector
from risk_engine import RiskEngine
from alerts import AlertSystem
from logger import EventLogger


def main():

    # ==================================
    # INITIALIZE MODULES
    # ==================================

    face_detector = FaceDetector()

    phone_detector = PhoneDetector()

    head_pose = HeadPoseEstimator()

    drowsiness_detector = DrowsinessDetector()

    risk_engine = RiskEngine()

    alert_system = AlertSystem()

    logger = EventLogger()

    # ==================================
    # CAMERA
    # ==================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    print("======================================")
    print("DriveGuard AI - Step 9G")
    print("Logger Integration")
    print("======================================")
    print("Events will be saved automatically.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read webcam.")
            break

        # ==================================
        # FACE DETECTION
        # ==================================

        face_result = face_detector.detect(frame)

        frame = face_detector.draw_landmarks(
            frame,
            face_result
        )

        # ==================================
        # PHONE DETECTION
        # ==================================

        phone_detected, phone_confidence, boxes = (
            phone_detector.detect(frame)
        )

        frame = phone_detector.draw_detections(
            frame,
            boxes
        )

        # ==================================
        # HEAD POSE
        # ==================================

        pose = head_pose.estimate(
            face_result
        )

        head_direction = pose["direction"]

        # ==================================
        # DROWSINESS
        # ==================================

        drowsiness = drowsiness_detector.detect(
            face_result
        )

        drowsy = drowsiness["drowsy"]

        # ==================================
        # RISK ENGINE
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
        # ALERT SYSTEM
        # ==================================

        alert_system.process_risk(risk)

        # ==================================
        # LOGGER
        # ==================================

        logger.log_event(
            risk_score=score,
            risk_level=level,
            phone_detected=phone_detected,
            head_direction=head_direction,
            drowsy=drowsy,
            events=events
        )

        # ==================================
        # DISPLAY
        # ==================================

        cv2.putText(
            frame,
            f"HEAD: {head_direction}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"PHONE: {'YES' if phone_detected else 'NO'}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"DROWSY: {'YES' if drowsy else 'NO'}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ==================================
        # RISK SCORE
        # ==================================

        if level == "LOW":

            risk_color = (0, 255, 0)

        elif level == "MEDIUM":

            risk_color = (0, 255, 255)

        else:

            risk_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"RISK SCORE: {score}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            risk_color,
            2
        )

        cv2.putText(
            frame,
            f"RISK LEVEL: {level}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            risk_color,
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
                "STATUS: NORMAL",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ==================================
        # LOG STATUS
        # ==================================

        cv2.putText(
            frame,
            "EVENT LOGGING: ON",
            (20, frame.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
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

        # ==================================
        # SHOW
        # ==================================

        cv2.imshow(
            "DriveGuard AI - Logger Integration",
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

    print("\n======================================")
    print("DriveGuard AI session ended.")
    print(
        f"Events saved to: {logger.get_log_file()}"
    )
    print("======================================")


if __name__ == "__main__":
    main()
