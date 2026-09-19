import pyttsx3
import time


class AlertSystem:

    def __init__(self):

        self.engine = pyttsx3.init()

        # Voice settings
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)

        # Prevent repeated alerts
        self.last_alert_time = 0
        self.alert_cooldown = 5

    def speak(self, message):

        current_time = time.time()

        # Don't repeat the same alert too quickly
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        print(f"[VOICE ALERT] {message}")

        self.engine.say(message)
        self.engine.runAndWait()

        self.last_alert_time = current_time

    def process_risk(self, risk_result):

        score = risk_result["score"]
        level = risk_result["level"]
        events = risk_result["events"]

        # No risk
        if level == "LOW":
            return

        # --------------------------------
        # HIGH RISK
        # --------------------------------

        if level == "HIGH":

            if "PHONE + HEAD DOWN" in events:

                self.speak(
                    "Warning. Put your phone down and look at the road."
                )

            elif "DROWSINESS" in events:

                self.speak(
                    "Warning. Driver drowsiness detected. Please take a break."
                )

            elif "PHONE USE" in events:

                self.speak(
                    "Warning. Please stop using your phone while driving."
                )

            elif "HEAD DOWN" in events:

                self.speak(
                    "Warning. Please keep your eyes and head forward."
                )

            else:

                self.speak(
                    "Warning. High driver distraction detected."
                )

        # --------------------------------
        # MEDIUM RISK
        # --------------------------------

        elif level == "MEDIUM":

            if "DROWSINESS" in events:

                self.speak(
                    "Alert. Signs of drowsiness detected."
                )

            elif "PHONE USE" in events:

                self.speak(
                    "Alert. Please avoid using your phone."
                )

            elif "HEAD AWAY" in events:

                self.speak(
                    "Alert. Please keep your attention forward."
                )

            else:

                self.speak(
                    "Alert. Driver distraction detected."
                )


def main():

    print("Testing DriveGuard AI Alert System...")

    alert_system = AlertSystem()

    # Test 1
    print("\nTest 1: Medium Risk - Phone")

    risk_result = {
        "score": 30,
        "level": "MEDIUM",
        "events": ["PHONE USE"]
    }

    alert_system.process_risk(risk_result)

    time.sleep(6)

    # Test 2
    print("\nTest 2: High Risk - Phone + Head Down")

    risk_result = {
        "score": 75,
        "level": "HIGH",
        "events": [
            "PHONE USE",
            "HEAD DOWN",
            "PHONE + HEAD DOWN"
        ]
    }

    alert_system.process_risk(risk_result)

    time.sleep(6)

    # Test 3
    print("\nTest 3: Drowsiness")

    risk_result = {
        "score": 40,
        "level": "MEDIUM",
        "events": ["DROWSINESS"]
    }

    alert_system.process_risk(risk_result)

    print("\nAlert system test completed.")


if __name__ == "__main__":
    main()
