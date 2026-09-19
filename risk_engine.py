import time


class RiskEngine:

    # Risk points
    PHONE_POINTS = 30
    HEAD_AWAY_POINTS = 20
    HEAD_DOWN_POINTS = 25
    DROWSY_POINTS = 40

    # Duration thresholds
    PHONE_DURATION = 1.0
    HEAD_AWAY_DURATION = 1.5
    HEAD_DOWN_DURATION = 1.0

    def __init__(self):

        self.phone_start = None
        self.head_away_start = None
        self.head_down_start = None

    def calculate_risk(
        self,
        phone_detected=False,
        head_direction="FORWARD",
        drowsy=False
    ):

        current_time = time.time()

        score = 0
        events = []

        # --------------------------------
        # PHONE DETECTION
        # --------------------------------

        if phone_detected:

            if self.phone_start is None:
                self.phone_start = current_time

            phone_duration = (
                current_time - self.phone_start
            )

            if phone_duration >= self.PHONE_DURATION:

                score += self.PHONE_POINTS
                events.append("PHONE USE")

        else:

            self.phone_start = None
            phone_duration = 0.0

        # --------------------------------
        # HEAD AWAY
        # --------------------------------

        if head_direction in ["LEFT", "RIGHT"]:

            if self.head_away_start is None:
                self.head_away_start = current_time

            head_away_duration = (
                current_time - self.head_away_start
            )

            if head_away_duration >= self.HEAD_AWAY_DURATION:

                score += self.HEAD_AWAY_POINTS
                events.append(
                    f"HEAD {head_direction}"
                )

        else:

            self.head_away_start = None
            head_away_duration = 0.0

        # --------------------------------
        # HEAD DOWN
        # --------------------------------

        if head_direction == "DOWN":

            if self.head_down_start is None:
                self.head_down_start = current_time

            head_down_duration = (
                current_time - self.head_down_start
            )

            if head_down_duration >= self.HEAD_DOWN_DURATION:

                score += self.HEAD_DOWN_POINTS
                events.append("HEAD DOWN")

        else:

            self.head_down_start = None
            head_down_duration = 0.0

        # --------------------------------
        # DROWSINESS
        # --------------------------------

        if drowsy:

            score += self.DROWSY_POINTS
            events.append("DROWSINESS")

        # --------------------------------
        # COMBINATION RISKS
        # --------------------------------

        if phone_detected and head_direction == "DOWN":

            score += 20
            events.append("PHONE + HEAD DOWN")

        if phone_detected and head_direction in ["LEFT", "RIGHT"]:

            score += 15
            events.append("PHONE + HEAD AWAY")

        if drowsy and head_direction in ["LEFT", "RIGHT"]:

            score += 15
            events.append("DROWSY + HEAD AWAY")

        # --------------------------------
        # LIMIT SCORE
        # --------------------------------

        score = min(score, 100)

        # --------------------------------
        # RISK LEVEL
        # --------------------------------

        if score < 30:

            level = "LOW"

        elif score < 60:

            level = "MEDIUM"

        else:

            level = "HIGH"

        return {
            "score": score,
            "level": level,
            "events": events,
            "phone_duration": phone_duration,
            "head_away_duration": head_away_duration,
            "head_down_duration": head_down_duration
        }
