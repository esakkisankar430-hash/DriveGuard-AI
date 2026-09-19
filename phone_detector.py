from ultralytics import YOLO
import cv2


class PhoneDetector:

    def __init__(self):
        # Load YOLO model
        self.model = YOLO("yolo11n.pt")

        # COCO class ID for cell phone
        self.phone_class_id = 67

    def detect(self, frame):

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        phone_detected = False
        confidence = 0.0
        boxes = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                if class_id == self.phone_class_id and conf >= 0.40:

                    phone_detected = True
                    confidence = max(confidence, conf)

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    boxes.append(
                        (x1, y1, x2, y2, conf)
                    )

        return phone_detected, confidence, boxes

    def draw_detections(self, frame, boxes):

        for x1, y1, x2, y2, conf in boxes:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"PHONE {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame


def main():

    # Test the detector independently
    detector = PhoneDetector()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        phone_detected, confidence, boxes = detector.detect(frame)

        frame = detector.draw_detections(
            frame,
            boxes
        )

        if phone_detected:

            cv2.putText(
                frame,
                f"PHONE DETECTED: {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO PHONE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "DriveGuard AI - Phone Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
