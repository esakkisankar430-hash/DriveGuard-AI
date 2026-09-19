import csv
import os
from datetime import datetime


class EventLogger:

    def __init__(self, log_directory="logs"):

        self.log_directory = log_directory

        # Create logs folder if it doesn't exist
        os.makedirs(
            self.log_directory,
            exist_ok=True
        )

        self.log_file = os.path.join(
            self.log_directory,
            "driveguard_events.csv"
        )

        # Create CSV file and header if it doesn't exist
        if not os.path.exists(self.log_file):

            with open(
                self.log_file,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Risk Score",
                    "Risk Level",
                    "Phone Detected",
                    "Head Direction",
                    "Drowsy",
                    "Events"
                ])

    def log_event(
        self,
        risk_score,
        risk_level,
        phone_detected,
        head_direction,
        drowsy,
        events
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        events_text = " | ".join(events)

        with open(
            self.log_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                risk_score,
                risk_level,
                "YES" if phone_detected else "NO",
                head_direction,
                "YES" if drowsy else "NO",
                events_text
            ])

    def get_log_file(self):

        return self.log_file


def main():

    print("Testing DriveGuard AI Logger...")

    logger = EventLogger()

    # Test 1 - Normal
    logger.log_event(
        risk_score=0,
        risk_level="LOW",
        phone_detected=False,
        head_direction="FORWARD",
        drowsy=False,
        events=[]
    )

    # Test 2 - Phone
    logger.log_event(
        risk_score=30,
        risk_level="MEDIUM",
        phone_detected=True,
        head_direction="FORWARD",
        drowsy=False,
        events=["PHONE USE"]
    )

    # Test 3 - Phone + Head Down
    logger.log_event(
        risk_score=75,
        risk_level="HIGH",
        phone_detected=True,
        head_direction="DOWN",
        drowsy=False,
        events=[
            "PHONE USE",
            "HEAD DOWN",
            "PHONE + HEAD DOWN"
        ]
    )

    # Test 4 - Drowsiness
    logger.log_event(
        risk_score=40,
        risk_level="MEDIUM",
        phone_detected=False,
        head_direction="FORWARD",
        drowsy=True,
        events=["DROWSINESS"]
    )

    print("\nLogger test completed.")
    print(
        f"Log file: {logger.get_log_file()}"
    )


if __name__ == "__main__":
    main()
