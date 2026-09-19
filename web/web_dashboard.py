import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# DRIVEGUARD AI MODULES
# ============================================================

from face_detector import FaceDetector
from phone_detector import PhoneDetector
from head_pose import HeadPoseEstimator
from drowsiness import DrowsinessDetector
from risk_engine import RiskEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 10px;
    }

    .risk-score {
        font-size: 42px;
        font-weight: 700;
    }

    .small-label {
        font-size: 14px;
        color: #777;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 DriveGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI-Based Driver Distraction & Drowsiness Monitoring</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "risk_history" not in st.session_state:
    st.session_state.risk_history = deque(maxlen=100)

if "event_count" not in st.session_state:
    st.session_state.event_count = 0


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class DriveGuardVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.face_detector = FaceDetector()

        self.phone_detector = PhoneDetector()

        self.head_pose = HeadPoseEstimator()

        self.drowsiness = DrowsinessDetector()

        self.risk_engine = RiskEngine()

        self.lock = None

        self.last_result = {
            "risk_score": 0,
            "risk_level": "LOW",
            "head_direction": "FORWARD",
            "phone_detected": False,
            "drowsy": False,
        }

        self.risk_history = deque(maxlen=100)

        self.last_time = time.time()

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        try:

            # ------------------------------------------------
            # FACE DETECTION
            # ------------------------------------------------

            face_result = self.face_detector.detect(image)

            # ------------------------------------------------
            # PHONE DETECTION
            # ------------------------------------------------

            phone_result = self.phone_detector.detect(image)

            phone_detected = False

            if phone_result:

                if isinstance(phone_result, list):

                    phone_detected = len(phone_result) > 0

                else:

                    phone_detected = bool(phone_result)

            # ------------------------------------------------
            # HEAD POSE
            # ------------------------------------------------

            head_result = self.head_pose.estimate(face_result)

            direction = "FORWARD"

            if head_result:

                direction = head_result.get(
                    "direction",
                    "FORWARD"
                )

            # ------------------------------------------------
            # DROWSINESS
            # ------------------------------------------------

            drowsy_result = self.drowsiness.detect(
                face_result
            )

            if isinstance(drowsy_result, tuple):

                is_drowsy = drowsy_result[0]

            elif isinstance(drowsy_result, dict):

                is_drowsy = drowsy_result.get(
                    "drowsy",
                    False
                )

            else:

                is_drowsy = bool(drowsy_result)

            # ------------------------------------------------
            # RISK ENGINE
            # ------------------------------------------------

            risk_result = self.risk_engine.calculate(
                phone_detected=phone_detected,
                head_direction=direction,
                drowsy=is_drowsy
            )

            risk_score = risk_result.get(
                "risk_score",
                risk_result.get("score", 0)
            )

            risk_level = risk_result.get(
                "risk_level",
                risk_result.get("level", "LOW")
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            self.last_result = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "head_direction": direction,
                "phone_detected": phone_detected,
                "drowsy": is_drowsy,
            }

            self.risk_history.append(risk_score)

            # ------------------------------------------------
            # DRAW INFORMATION
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (10, 10),
                (470, 185),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                image,
                f"Risk Score: {risk_score}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                f"Risk Level: {risk_level}",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                f"Head: {direction}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                f"Phone: {'DETECTED' if phone_detected else 'CLEAR'}",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                f"Drowsiness: {'YES' if is_drowsy else 'NO'}",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        except Exception as e:

            cv2.putText(
                image,
                "Processing Error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ DriveGuard AI")

    st.write(
        "Browser-based public demonstration "
        "of the DriveGuard AI prototype."
    )

    st.divider()

    st.subheader("Detection Modules")

    st.write("✅ Face / Facial Landmarks")
    st.write("✅ Phone Detection")
    st.write("✅ Head Pose Estimation")
    st.write("✅ Drowsiness Detection")
    st.write("✅ Risk Assessment")

    st.divider()

    st.caption(
        "Prototype only — do not use while actually driving."
    )


# ============================================================
# CAMERA
# ============================================================

st.subheader("📷 Live Driver Monitoring")

st.info(
    "Click START and allow your browser to access the camera."
)

ctx = webrtc_streamer(
    key="driveguard-camera",
    video_processor_factory=DriveGuardVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# ============================================================
# INFORMATION PANEL
# ============================================================

st.divider()

st.subheader("📊 Real-Time Analysis")

col1, col2, col3, col4, col5 = st.columns(5)

if ctx.video_processor:

    result = ctx.video_processor.last_result

    risk_score = result["risk_score"]

    risk_level = result["risk_level"]

    head_direction = result["head_direction"]

    phone_detected = result["phone_detected"]

    drowsy = result["drowsy"]

else:

    risk_score = 0

    risk_level = "LOW"

    head_direction = "FORWARD"

    phone_detected = False

    drowsy = False


with col1:

    st.metric(
        "Risk Score",
        risk_score
    )


with col2:

    st.metric(
        "Risk Level",
        risk_level
    )


with col3:

    st.metric(
        "Head Direction",
        head_direction
    )


with col4:

    st.metric(
        "Phone",
        "DETECTED" if phone_detected else "CLEAR"
    )


with col5:

    st.metric(
        "Drowsiness",
        "YES" if drowsy else "NO"
    )


# ============================================================
# WARNING
# ============================================================

if risk_level == "HIGH":

    st.error(
        "🚨 HIGH RISK — Multiple unsafe driver behaviours detected."
    )

elif risk_level == "MEDIUM":

    st.warning(
        "⚠️ MEDIUM RISK — Driver attention required."
    )

else:

    st.success(
        "✅ LOW RISK — No significant unsafe behaviour detected."
    )


# ============================================================
# RISK HISTORY
# ============================================================

if ctx.video_processor:

    history = list(
        ctx.video_processor.risk_history
    )

    if history:

        st.subheader("📈 Risk Score History")

        st.line_chart(history)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.subheader("🔍 Detection Status")

c1, c2, c3 = st.columns(3)

with c1:

    st.write(
        "### 📱 Phone Detection"
    )

    if phone_detected:

        st.error("Phone detected")

    else:

        st.success("No phone detected")


with c2:

    st.write(
        "### 👤 Head Pose"
    )

    st.info(
        f"Direction: {head_direction}"
    )


with c3:

    st.write(
        "### 😴 Drowsiness"
    )

    if drowsy:

        st.error("Drowsiness detected")

    else:

        st.success("Alert")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DriveGuard AI | Computer Vision Based Driver Monitoring Prototype"
)

st.caption(
    "For academic demonstration and research purposes only."
)
