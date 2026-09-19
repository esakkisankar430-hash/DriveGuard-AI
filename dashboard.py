import streamlit as st
import cv2
import time
from collections import deque
import pandas as pd

from face_detector import FaceDetector
from phone_detector import PhoneDetector
from head_pose import HeadPoseEstimator
from drowsiness import DrowsinessDetector
from risk_engine import RiskEngine
from alerts import AlertSystem
from logger import EventLogger


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b0f14;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 20px;
}

.metric-card {
    background: #111827;
    border: 1px solid #263241;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    min-height: 120px;
}

.metric-title {
    color: #9ca3af;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.risk-card {
    background: #111827;
    border: 1px solid #263241;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.risk-score {
    font-size: 64px;
    font-weight: 800;
}

.low {
    color: #22c55e;
}

.medium {
    color: #f59e0b;
}

.high {
    color: #ef4444;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
}

.status-box {
    background: #111827;
    border-radius: 12px;
    border: 1px solid #263241;
    padding: 15px;
    margin-bottom: 8px;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "camera" not in st.session_state:
    st.session_state.camera = None

if "face_detector" not in st.session_state:
    st.session_state.face_detector = None

if "phone_detector" not in st.session_state:
    st.session_state.phone_detector = None

if "head_pose" not in st.session_state:
    st.session_state.head_pose = None

if "drowsiness" not in st.session_state:
    st.session_state.drowsiness = None

if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = None

if "alerts" not in st.session_state:
    st.session_state.alerts = None

if "logger" not in st.session_state:
    st.session_state.logger = None

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "risk_history" not in st.session_state:
    st.session_state.risk_history = deque(maxlen=100)

if "event_history" not in st.session_state:
    st.session_state.event_history = deque(maxlen=30)

if "event_counts" not in st.session_state:
    st.session_state.event_counts = {
        "PHONE USE": 0,
        "HEAD DOWN": 0,
        "HEAD LEFT": 0,
        "HEAD RIGHT": 0,
        "DROWSINESS": 0
    }

if "last_result" not in st.session_state:
    st.session_state.last_result = {
        "score": 0,
        "level": "LOW",
        "events": [],
        "head_direction": "FORWARD",
        "phone_detected": False,
        "drowsy": False
    }


# ============================================================
# START SYSTEM
# ============================================================

def start_system():

    # Open webcam
    camera = cv2.VideoCapture(0)

    # Check camera
    if not camera.isOpened():
        st.error(
            "❌ Camera could not be opened. "
            "Make sure your webcam is connected and not being used by another application."
        )
        return False

    # Camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Store camera
    st.session_state.camera = camera

    # Initialize AI modules
    st.session_state.face_detector = FaceDetector()
    st.session_state.phone_detector = PhoneDetector()
    st.session_state.head_pose = HeadPoseEstimator()
    st.session_state.drowsiness = DrowsinessDetector()
    st.session_state.risk_engine = RiskEngine()
    st.session_state.alerts = AlertSystem()
    st.session_state.logger = EventLogger()

    # Start time
    st.session_state.start_time = time.time()

    # Clear previous session data
    st.session_state.risk_history.clear()
    st.session_state.event_history.clear()

    st.session_state.event_counts = {
        "PHONE USE": 0,
        "HEAD DOWN": 0,
        "HEAD LEFT": 0,
        "HEAD RIGHT": 0,
        "DROWSINESS": 0
    }

    st.session_state.last_result = {
        "score": 0,
        "level": "LOW",
        "events": [],
        "head_direction": "FORWARD",
        "phone_detected": False,
        "drowsy": False
    }

    st.session_state.running = True

    return True


# ============================================================
# STOP SYSTEM
# ============================================================

def stop_system():

    if st.session_state.camera is not None:
        st.session_state.camera.release()

    st.session_state.camera = None

    st.session_state.running = False


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    face_detector = st.session_state.face_detector
    phone_detector = st.session_state.phone_detector
    head_pose = st.session_state.head_pose
    drowsiness = st.session_state.drowsiness
    risk_engine = st.session_state.risk_engine
    alerts = st.session_state.alerts
    logger = st.session_state.logger

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_result = face_detector.detect(frame)

    frame = face_detector.draw_landmarks(
        frame,
        face_result
    )

    # --------------------------------------------------------
    # PHONE DETECTION
    # --------------------------------------------------------

    phone_detected, phone_confidence, phone_boxes = (
        phone_detector.detect(frame)
    )

    frame = phone_detector.draw_detections(
        frame,
        phone_boxes
    )

    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    head_result = head_pose.estimate(
        face_result
    )

    direction = head_result["direction"]

    # --------------------------------------------------------
    # DROWSINESS
    # --------------------------------------------------------

    drowsy_result = drowsiness.detect(
        face_result
    )

    is_drowsy = drowsy_result["drowsy"]

    # --------------------------------------------------------
    # RISK ENGINE
    # --------------------------------------------------------

    risk_result = risk_engine.calculate_risk(
        phone_detected=phone_detected,
        head_direction=direction,
        drowsy=is_drowsy
    )

    score = risk_result["score"]
    level = risk_result["level"]
    events = risk_result["events"]

    # --------------------------------------------------------
    # ALERT
    # --------------------------------------------------------

    alerts.process_risk(
        risk_result
    )

    # --------------------------------------------------------
    # LOGGER
    # --------------------------------------------------------

    logger.log_event(
        risk_score=score,
        risk_level=level,
        phone_detected=phone_detected,
        head_direction=direction,
        drowsy=is_drowsy,
        events=events
    )

    # --------------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------------

    st.session_state.risk_history.append({
        "Time": time.strftime("%H:%M:%S"),
        "Risk": score
    })

    for event in events:

        if event in st.session_state.event_counts:
            st.session_state.event_counts[event] += 1

        elif "HEAD LEFT" in event:
            st.session_state.event_counts["HEAD LEFT"] += 1

        elif "HEAD RIGHT" in event:
            st.session_state.event_counts["HEAD RIGHT"] += 1

        elif "HEAD DOWN" in event:
            st.session_state.event_counts["HEAD DOWN"] += 1

        elif "PHONE" in event:
            st.session_state.event_counts["PHONE USE"] += 1

        elif "DROWSY" in event or "DROWSINESS" in event:
            st.session_state.event_counts["DROWSINESS"] += 1

        st.session_state.event_history.appendleft(
            f"{time.strftime('%H:%M:%S')} — {event}"
        )

    # --------------------------------------------------------
    # STORE COMPLETE RESULT
    # --------------------------------------------------------

    risk_result["head_direction"] = direction
    risk_result["phone_detected"] = phone_detected
    risk_result["drowsy"] = is_drowsy

    st.session_state.last_result = risk_result

    # --------------------------------------------------------
    # CAMERA OVERLAY
    # --------------------------------------------------------

    if level == "HIGH":
        risk_color = (0, 0, 255)

    elif level == "MEDIUM":
        risk_color = (0, 165, 255)

    else:
        risk_color = (0, 200, 0)

    # --------------------------------------------------------
    # RISK BOX
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 155),
        (15, 20, 28),
        -1
    )

    cv2.putText(
        frame,
        f"RISK: {score}/100",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        risk_color,
        3
    )

    cv2.putText(
        frame,
        f"LEVEL: {level}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        risk_color,
        2
    )

    cv2.putText(
        frame,
        f"HEAD: {direction}",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # PHONE WARNING
    # --------------------------------------------------------

    if phone_detected:

        cv2.putText(
            frame,
            "PHONE DETECTED",
            (450, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # DROWSINESS WARNING
    # --------------------------------------------------------

    if is_drowsy:

        cv2.putText(
            frame,
            "DROWSINESS DETECTED",
            (450, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return frame


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🚗 DriveGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Driver Distraction & Drowsiness Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System Control")

    # --------------------------------------------------------
    # START / STOP
    # --------------------------------------------------------

    if not st.session_state.running:

        if st.button(
            "▶ START MONITORING",
            width="stretch"
        ):

            if start_system():
                st.rerun()

    else:

        if st.button(
            "⏹ STOP MONITORING",
            width="stretch"
        ):

            stop_system()
            st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # MODULE STATUS
    # --------------------------------------------------------

    st.markdown("### System Modules")

    st.write("🟢 Face Detection")
    st.write("🟢 Phone Detection")
    st.write("🟢 Head Pose")
    st.write("🟢 Drowsiness Detection")
    st.write("🟢 Risk Engine")
    st.write("🟢 Voice Alerts")
    st.write("🟢 Event Logger")

    st.markdown("---")

    st.caption(
        "Prototype system — test only while stationary. "
        "Do not use while actually driving."
    )


# ============================================================
# STANDBY DASHBOARD
# ============================================================

if not st.session_state.running:

    st.info(
        "Click **START MONITORING** from the sidebar to begin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
            <div class="metric-title">AI MODEL</div>
            <div class="metric-value">ACTIVE</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
            <div class="metric-title">CAMERA</div>
            <div class="metric-value">READY</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
            <div class="metric-title">STATUS</div>
            <div class="metric-value">STANDBY</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LIVE DASHBOARD
# ============================================================

if st.session_state.running:

    @st.fragment(run_every=0.5)
    def live_dashboard():

        camera = st.session_state.camera

        # ----------------------------------------------------
        # CAMERA CHECK
        # ----------------------------------------------------

        if camera is None:

            st.error("❌ Camera object is not available.")

            return

        if not camera.isOpened():

            st.error(
                "❌ Camera could not be opened. "
                "Close other applications using the webcam."
            )

            return

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = camera.read()

        if not ret:

            st.error(
                "❌ Unable to read camera frame."
            )

            return

        # ----------------------------------------------------
        # MIRROR FRAME
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # PROCESS FRAME
        # ----------------------------------------------------

        processed_frame = process_frame(
            frame
        )

        # ----------------------------------------------------
        # GET CURRENT RESULT
        # ----------------------------------------------------

        result = st.session_state.last_result

        score = result["score"]

        level = result["level"]

        direction = result.get(
            "head_direction",
            "FORWARD"
        )

        phone_detected = result.get(
            "phone_detected",
            False
        )

        drowsy = result.get(
            "drowsy",
            False
        )

        # ----------------------------------------------------
        # RISK CLASS
        # ----------------------------------------------------

        if level == "LOW":

            risk_class = "low"

        elif level == "MEDIUM":

            risk_class = "medium"

        else:

            risk_class = "high"

        # ----------------------------------------------------
        # SESSION TIME
        # ----------------------------------------------------

        elapsed = 0

        if st.session_state.start_time:

            elapsed = int(
                time.time()
                -
                st.session_state.start_time
            )

        minutes = elapsed // 60

        seconds = elapsed % 60

        # ====================================================
        # TOP METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        with col1:

            st.markdown(
                f"""
                <div class="risk-card">
                <div class="metric-title">RISK SCORE</div>
                <div class="risk-score {risk_class}">
                {score}
                </div>
                <div>{level}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # HEAD DIRECTION
        # ----------------------------------------------------

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">HEAD DIRECTION</div>
                <div class="metric-value">
                {direction}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # PHONE
        # ----------------------------------------------------

        with col3:

            phone_status = (
                "DETECTED"
                if phone_detected
                else
                "CLEAR"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">PHONE STATUS</div>
                <div class="metric-value">
                {phone_status}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # SESSION TIME
        # ----------------------------------------------------

        with col4:

            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">SESSION TIME</div>
                <div class="metric-value">
                {minutes:02d}:{seconds:02d}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # LIVE CAMERA
        # ====================================================

        st.markdown(
            '<div class="section-title">📹 Live Camera Feed</div>',
            unsafe_allow_html=True
        )

        # Convert BGR → RGB
        display_frame = cv2.cvtColor(
            processed_frame,
            cv2.COLOR_BGR2RGB
        )

        # Display frame
        st.image(
            display_frame,
            width="stretch",
            output_format="JPEG"
        )

        # ====================================================
        # DETECTION STATUS
        # ====================================================

        st.markdown(
            '<div class="section-title">🔍 Detection Status</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # PHONE STATUS
        # ----------------------------------------------------

        with col1:

            if phone_detected:

                st.error(
                    "📱 Phone: DETECTED"
                )

            else:

                st.success(
                    "📱 Phone: CLEAR"
                )

        # ----------------------------------------------------
        # DROWSINESS STATUS
        # ----------------------------------------------------

        with col2:

            if drowsy:

                st.error(
                    "😴 Eyes: DROWSY"
                )

            else:

                st.success(
                    "👁️ Eyes: NORMAL"
                )

        # ----------------------------------------------------
        # HEAD STATUS
        # ----------------------------------------------------

        with col3:

            st.info(
                f"🧭 Head: {direction}"
            )

        # ====================================================
        # RISK GRAPH
        # ====================================================

        st.markdown(
            '<div class="section-title">📈 Risk History</div>',
            unsafe_allow_html=True
        )

        if len(
            st.session_state.risk_history
        ) > 1:

            graph_df = pd.DataFrame(
                list(
                    st.session_state.risk_history
                )
            )

            graph_df = graph_df.set_index(
                "Time"
            )

            st.line_chart(
                graph_df["Risk"],
                height=250
            )

        else:

            st.caption(
                "Collecting risk history..."
            )

        # ====================================================
        # EVENT COUNTERS
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Event Statistics</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        counts = st.session_state.event_counts

        with c1:

            st.metric(
                "📱 Phone",
                counts["PHONE USE"]
            )

        with c2:

            st.metric(
                "⬇️ Head Down",
                counts["HEAD DOWN"]
            )

        with c3:

            st.metric(
                "⬅️ Head Left",
                counts["HEAD LEFT"]
            )

        with c4:

            st.metric(
                "➡️ Head Right",
                counts["HEAD RIGHT"]
            )

        with c5:

            st.metric(
                "😴 Drowsiness",
                counts["DROWSINESS"]
            )

        # ====================================================
        # EVENT TIMELINE
        # ====================================================

        st.markdown(
            '<div class="section-title">🕒 Live Event Timeline</div>',
            unsafe_allow_html=True
        )

        if st.session_state.event_history:

            for event in list(
                st.session_state.event_history
            )[:10]:

                st.write(
                    f"🔴 {event}"
                )

        else:

            st.success(
                "No distraction events detected."
            )

        # ====================================================
        # LOG STATUS
        # ====================================================

        st.markdown(
            '<div class="section-title">💾 System Log</div>',
            unsafe_allow_html=True
        )

        if st.session_state.logger:

            st.success(
                "Event logging active"
            )

            st.caption(
                st.session_state.logger.get_log_file()
            )


    # Run live dashboard
    live_dashboard()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    DriveGuard AI • Intelligent Driver Safety Monitoring Prototype
    </div>
    """,
    unsafe_allow_html=True
)
