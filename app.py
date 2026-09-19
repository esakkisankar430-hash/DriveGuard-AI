import cv2
import time

from face_detector import FaceDetector
from phone_detector import PhoneDetector
from head_pose import HeadPoseEstimator
from drowsiness import DrowsinessDetector
from risk_engine import RiskEngine
from alerts import AlertSystem
from logger import EventLogger


class DriveGuardApp:

    def __init__(self):

        print("========================================")
        print("        DRIVEGUARD AI INITIALIZING")
        print("========================================")

        # ----------------------------------
        # AI MODULES
        # ----------------------------------

        self.face_detector = FaceDetector()

        self.phone_detector = PhoneDetector()

        self.head_pose = HeadPoseEstimator()

        self.drowsiness_detector = (
            DrowsinessDetector()
        )

        self.risk_engine = RiskEngine()

        self.alert_system = AlertSystem()

        self.logger = EventLogger()

        # ----------------------------------
        # CAMERA
        # ----------------------------------

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )

        # Try to use a reasonable camera resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        # ----------------------------------
        # SESSION STATISTICS
        # ----------------------------------

        self.start_time = time.time()

        self.frame_count = 0

        self.high_risk_count = 0

        self.medium_risk_count = 0

        self.phone_event_count = 0

        self.drowsiness_event_count = 0

        self.head_away_event_count = 0

        # Prevent counting the same event every frame
        self.previous_events = set()

        self.previous_level = "LOW"

        print("All modules loaded successfully.")
        print("Camera started.")
        print("Press Q to quit.")

    # ======================================
    # EVENT STATISTICS
    # ======================================

    def update_statistics(self, risk):

        events = set(risk["events"])

        level = risk["level"]

        # ----------------------------------
        # Risk level changes
        # ----------------------------------

        if level == "HIGH" and self.previous_level != "HIGH":

            self.high_risk_count += 1

        elif (
            level == "MEDIUM"
            and self.previous_level == "LOW"
        ):

            self.medium_risk_count += 1

        self.previous_level = level

        # ----------------------------------
        # New events only
        # ----------------------------------

        new_events = (
            events - self.previous_events
        )

        for event in new_events:

            if "PHONE" in event:

                self.phone_event_count += 1

            if "DROWSINESS" in event:

                self.drowsiness_event_count += 1

            if (
                "HEAD LEFT" in event
                or "HEAD RIGHT" in event
            ):

                self.head_away_event_count += 1

        self.previous_events = events

    # ======================================
    # DRAW DASHBOARD
    # ======================================

    def draw_dashboard(
        self,
        frame,
        head_direction,
        phone_detected,
        phone_confidence,
        drowsy,
        ear,
        risk,
        session_time
    ):

        height, width = frame.shape[:2]

        score = risk["score"]

        level = risk["level"]

        events = risk["events"]

        # ==================================
        # TOP HEADER
        # ==================================

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 65),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "DRIVEGUARD AI",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "REAL-TIME DRIVER MONITORING",
            (width - 430, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2
        )

        # ==================================
        # LEFT INFORMATION PANEL
        # ==================================

        panel_x = 15
        panel_y = 85

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (360, 350),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "DRIVER STATUS",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # Head
        cv2.putText(
            frame,
            f"Head Direction : {head_direction}",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Phone
        phone_text = "DETECTED" if phone_detected else "NOT DETECTED"

        cv2.putText(
            frame,
            f"Phone         : {phone_text}",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if phone_detected:

            cv2.putText(
                frame,
                f"Confidence    : {phone_confidence:.2f}",
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1
            )

        # Drowsiness
        drowsy_text = "YES" if drowsy else "NO"

        cv2.putText(
            frame,
            f"Drowsiness    : {drowsy_text}",
            (30, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # EAR
        cv2.putText(
            frame,
            f"Eye EAR       : {ear:.3f}",
            (30, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Session
        cv2.putText(
            frame,
            f"Session       : {session_time:.0f}s",
            (30, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ==================================
        # RISK PANEL
        # ==================================

        risk_x = width - 330

        cv2.rectangle(
            frame,
            (risk_x, 85),
            (width - 15, 300),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "RISK MONITOR",
            (risk_x + 20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # Risk color
        if level == "LOW":

            risk_color = (0, 255, 0)

        elif level == "MEDIUM":

            risk_color = (0, 255, 255)

        else:

            risk_color = (0, 0, 255)

        # Score
        cv2.putText(
            frame,
            str(score),
            (risk_x + 100, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            risk_color,
            4
        )

        cv2.putText(
            frame,
            "/ 100",
            (risk_x + 175, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2
        )

        # Level
        cv2.putText(
            frame,
            f"LEVEL: {level}",
            (risk_x + 45, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            risk_color,
            2
        )

        # ==================================
        # EVENT PANEL
        # ==================================

        event_y = 375

        cv2.rectangle(
            frame,
            (15, event_y),
            (width - 15, height - 60),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "CURRENT EVENTS",
            (30, event_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        if events:

            y = event_y + 75

            for event in events:

                cv2.putText(
                    frame,
                    f"!  {event}",
                    (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                y += 32

        else:

            cv2.putText(
                frame,
                "No active risk events",
                (35, event_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ==================================
        # FOOTER
        # ==================================

        cv2.rectangle(
            frame,
            (0, height - 45),
            (width, height),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "EVENT LOGGING: ON",
            (20, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Q = EXIT",
            (width - 100, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        return frame

    # ======================================
    # MAIN LOOP
    # ======================================

    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:

                print("Could not read webcam frame.")

                break

            self.frame_count += 1

            # ==================================
            # FACE
            # ==================================

            face_result = self.face_detector.detect(
                frame
            )

            frame = self.face_detector.draw_landmarks(
                frame,
                face_result
            )

            # ==================================
            # PHONE
            # ==================================

            (
                phone_detected,
                phone_confidence,
                phone_boxes
            ) = self.phone_detector.detect(frame)

            frame = self.phone_detector.draw_detections(
                frame,
                phone_boxes
            )

            # ==================================
            # HEAD POSE
            # ==================================

            pose = self.head_pose.estimate(
                face_result
            )

            head_direction = pose["direction"]

            # ==================================
            # DROWSINESS
            # ==================================

            drowsiness = (
                self.drowsiness_detector.detect(
                    face_result
                )
            )

            drowsy = drowsiness["drowsy"]

            ear = drowsiness["ear"]

            # ==================================
            # RISK ENGINE
            # ==================================

            risk = self.risk_engine.calculate_risk(
                phone_detected=phone_detected,
                head_direction=head_direction,
                drowsy=drowsy
            )

            # ==================================
            # ALERT
            # ==================================

            self.alert_system.process_risk(
                risk
            )

            # ==================================
            # LOG EVENT
            # ==================================

            self.logger.log_event(
                risk_score=risk["score"],
                risk_level=risk["level"],
                phone_detected=phone_detected,
                head_direction=head_direction,
                drowsy=drowsy,
                events=risk["events"]
            )

            # ==================================
            # STATISTICS
            # ==================================

            self.update_statistics(risk)

            session_time = (
                time.time() - self.start_time
            )

            # ==================================
            # DASHBOARD
            # ==================================

            frame = self.draw_dashboard(
                frame,
                head_direction,
                phone_detected,
                phone_confidence,
                drowsy,
                ear,
                risk,
                session_time
            )

            # ==================================
            # SHOW
            # ==================================

            cv2.imshow(
                "DriveGuard AI",
                frame
            )

            # ==================================
            # QUIT
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

        self.shutdown()

    # ======================================
    # SHUTDOWN
    # ======================================

    def shutdown(self):

        print("\n========================================")
        print("       DRIVEGUARD AI SESSION ENDED")
        print("========================================")

        self.cap.release()

        self.face_detector.close()

        cv2.destroyAllWindows()

        session_time = (
            time.time() - self.start_time
        )

        print(
            f"Session duration : {session_time:.1f} seconds"
        )

        print(
            f"Frames processed : {self.frame_count}"
        )

        print(
            f"High-risk events : {self.high_risk_count}"
        )

        print(
            f"Medium-risk events : {self.medium_risk_count}"
        )

        print(
            f"Phone events : {self.phone_event_count}"
        )

        print(
            f"Drowsiness events : "
            f"{self.drowsiness_event_count}"
        )

        print(
            f"Head-away events : "
            f"{self.head_away_event_count}"
        )

        print(
            f"Log file : {self.logger.get_log_file()}"
        )

        print("========================================")


def main():

    try:

        app = DriveGuardApp()

        app.run()

    except Exception as error:

        print("\nERROR:")
        print(error)

        print(
            "\nCheck that all DriveGuard AI modules "
            "and model files are present."
        )


if __name__ == "__main__":

    main()
