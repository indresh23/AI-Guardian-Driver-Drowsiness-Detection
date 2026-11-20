import os
import sys
import math
import time
import threading
import urllib.request
import requests
import cv2
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Guardian — Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# OUTFIT GOOGLE FONT & GLASSMORPHISM / ANIMATION CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, button, input, select, textarea {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617 75%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        gap: 0.85rem;
    }

    /* Centered Glass Login Box */
    .login-box {
        max-width: 440px;
        margin: 4rem auto 1.5rem auto;
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
        transition: all 0.3s ease;
    }

    /* Glass Panel Metric Cards */
    .glass-panel {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .glass-panel:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Status Cards & Transitions */
    .status-card {
        padding: 15px 20px;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.03em;
        margin-bottom: 14px;
        transition: all 0.35s ease-in-out;
    }
    .status-idle {
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: #cbd5e1;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.55);
        color: #34d399;
    }

    @keyframes pulse-border {
        0% {
            border-color: rgba(239, 68, 68, 0.5);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75);
        }
        50% {
            border-color: rgba(248, 113, 113, 1);
            box-shadow: 0 0 24px 6px rgba(239, 68, 68, 0.5);
        }
        100% {
            border-color: rgba(239, 68, 68, 0.5);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75);
        }
    }

    .status-warning {
        background: rgba(239, 68, 68, 0.25);
        border: 2px solid #ef4444;
        color: #fecaca;
        animation: pulse-border 1.5s infinite ease-in-out;
    }

    /* Glowing Rest Stop Card */
    .rest-stop-card {
        background: rgba(245, 158, 11, 0.14);
        border: 1.5px solid rgba(251, 191, 36, 0.75);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.28);
        border-radius: 16px;
        padding: 18px;
        margin-top: 12px;
        color: #fde68a;
        transition: all 0.3s ease;
    }
    .emergency-btn {
        display: inline-block;
        width: 100%;
        text-align: center;
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: #ffffff !important;
        font-weight: 700;
        padding: 10px 14px;
        border-radius: 10px;
        text-decoration: none !important;
        margin-top: 10px;
        border: 1px solid rgba(254, 202, 202, 0.4);
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4);
        transition: transform 0.2s ease;
    }
    .emergency-btn:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "driver_name" not in st.session_state:
    st.session_state.driver_name = "Driver"
if "emergency_phone" not in st.session_state:
    st.session_state.emergency_phone = "+91112"
if "run_system" not in st.session_state:
    st.session_state.run_system = False

# ---------------------------------------------------------
# GLASSMORPHISM LOGIN GATE
# ---------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="login-box">
            <h2 style="margin-top:0; color:#f8fafc;">🛡️ AI Guardian</h2>
            <p style="color:#94a3b8; font-size:0.95rem; margin-bottom:0;">
                Driver Drowsiness & Fatigue Monitoring System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center_col, _ = st.columns([1, 1.1, 1])
    with center_col:
        with st.form("driver_login_form"):
            driver_input = st.text_input("Driver Name / ID", value="Lakshya")
            emergency_input = st.text_input("Emergency Contact Number", value="+91112")
            passcode = st.text_input("Access PIN (any 4 digits)", type="password", value="1234")
            submitted = st.form_submit_button("Launch AI Guardian", width="stretch", type="primary")
            if submitted:
                if driver_input.strip() and len(passcode.strip()) >= 4:
                    st.session_state.authenticated = True
                    st.session_state.driver_name = driver_input.strip()
                    st.session_state.emergency_phone = emergency_input.strip() or "+91112"
                    st.rerun()
                else:
                    st.error("Please enter a Driver Name and at least a 4-character PIN.")
    st.stop()

# ---------------------------------------------------------
# CONSTANTS & MEDIAPIPE FACE LANDMARKER SETUP
# ---------------------------------------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
TEMP_VIDEO_FILE = "temp_vid.mp4"

# MediaPipe 478-point FaceMesh Eye Landmark Indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading MediaPipe Face Landmarker model (~30 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@st.cache_resource
def load_face_landmarker():
    ensure_model_downloaded()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------
# EAR, BACKGROUND AUDIO ALERT & LIVE REST STOPS
# ---------------------------------------------------------
def compute_ear(eye_points: np.ndarray) -> float:
    vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
    vertical_2 = dist.euclidean(eye_points[2], eye_points[4])
    horizontal = dist.euclidean(eye_points[0], eye_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _play_beep_worker():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1200, 350)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def async_beep():
    thread = threading.Thread(target=_play_beep_worker, daemon=True)
    thread.start()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


@st.cache_data(ttl=600)
def get_live_rest_stops():
    try:
        geo = requests.get("https://ipinfo.io/json", timeout=3).json()
        lat, lon = map(float, geo["loc"].split(","))
        city = geo.get("city", "your area")

        query = f"""
        [out:json][timeout:6];
        (
          node["amenity"~"fuel|cafe|fast_food"]["name"](around:6000,{lat},{lon});
          way["amenity"~"fuel|cafe|fast_food"]["name"](around:6000,{lat},{lon});
          node["highway"="rest_area"](around:6000,{lat},{lon});
        );
        out center 10;
        """
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=6,
        ).json()

        places = []
        seen_names = set()
        for el in resp.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "Highway Rest Area")
            if name in seen_names:
                continue
            seen_names.add(name)

            p_lat = el.get("lat") or el.get("center", {}).get("lat")
            p_lon = el.get("lon") or el.get("center", {}).get("lon")
            if p_lat is None or p_lon is None:
                continue

            dist_km = haversine_km(lat, lon, float(p_lat), float(p_lon))
            kind = (
                tags.get("amenity", tags.get("highway", "Stop"))
                .replace("_", " ")
                .title()
            )
            places.append((dist_km, name, kind))

        places.sort(key=lambda x: x[0])
        top_places = places[:3]

        if top_places:
            items_html = "".join(
                f"<li><b>{name}</b> ({kind}) — {d:.1f} km away</li>"
                for d, name, kind in top_places
            )
            maps_link = f"https://www.google.com/maps/search/rest+stop+fuel+cafe/@{lat},{lon},14z"
            return items_html, maps_link, city
    except Exception:
        pass

    return (
        "<li><b>Nearest Fuel & Coffee Stop</b> — Open Maps for live routing</li>"
        "<li><b>Highway Safe Parking Bay</b> — Open Maps for live routing</li>",
        "https://www.google.com/maps/search/rest+stop+near+me",
        "your location",
    )


# ---------------------------------------------------------
# SIDEBAR — AI GUARDIAN CONTROLS
# ---------------------------------------------------------
st.sidebar.markdown("## 🛡️ AI Guardian")
st.sidebar.caption(f"Operator: **{st.session_state.driver_name}**")
st.sidebar.divider()

st.sidebar.markdown("### 🎛️ Sensitivity Parameters")
ear_threshold = st.sidebar.slider(
    "EAR Threshold",
    min_value=0.15,
    max_value=0.35,
    value=0.25,
    step=0.01,
    help="Eye Aspect Ratio cutoff. Lower = stricter.",
)

warning_frames = st.sidebar.slider(
    "Warning Frames",
    min_value=5,
    max_value=60,
    value=20,
    step=1,
    help="Consecutive closed-eye frames before alert.",
)

st.sidebar.divider()
video_source = st.sidebar.radio(
    "Video Input Source",
    options=["Webcam", "Upload Video"],
    index=0,
)

uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Driver Video", type=["mp4", "avi", "mov", "mkv"]
    )

st.sidebar.divider()
col_btn1, col_btn2 = st.sidebar.columns(2)
start_scan_btn = col_btn1.button("▶️ Start Scan", width="stretch", type="primary")
standby_btn = col_btn2.button("⏸️ Standby", width="stretch")

if start_scan_btn:
    st.session_state.run_system = True
if standby_btn:
    st.session_state.run_system = False

st.sidebar.markdown("")
if st.sidebar.button("🚪 Logout", width="stretch"):
    st.session_state.run_system = False
    st.session_state.authenticated = False
    st.rerun()

# ---------------------------------------------------------
# MAIN DASHBOARD & METRICS RENDERER
# ---------------------------------------------------------
st.title("🚗 AI Guardian — Driver Drowsiness Monitor")
st.caption("Real-time facial landmark tracking, EAR fatigue scoring & emergency rest-stop guidance")

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
fps_placeholder = m_col1.empty()
ear_placeholder = m_col2.empty()
score_placeholder = m_col3.empty()
blink_placeholder = m_col4.empty()
alert_placeholder = m_col5.empty()


def update_metrics(fps: float, ear: float, drowsy_score: int, blinks: int, alerts: int):
    ear_color = "#f87171" if (0 < ear < ear_threshold) else "#34d399"
    score_color = "#f87171" if drowsy_score >= 70 else ("#fbbf24" if drowsy_score >= 35 else "#38bdf8")

    cards = [
        (fps_placeholder, "Live FPS", f"{fps:.1f}", "#38bdf8"),
        (ear_placeholder, "Live EAR", f"{ear:.2f}", ear_color),
        (score_placeholder, "Drowsy Score", f"{drowsy_score}%", score_color),
        (blink_placeholder, "Blink Count", str(blinks), "#38bdf8"),
        (alert_placeholder, "Alert Count", str(alerts), "#f87171"),
    ]
    for placeholder, label, val, color in cards:
        placeholder.markdown(
            f"""
            <div class="glass-panel">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: {color};">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Initialize idle metrics on load / standby
update_metrics(0.0, 0.0, 0, 0, 0)

status_placeholder = st.empty()
status_placeholder.markdown(
    '<div class="status-card status-idle">⚪ SYSTEM STANDBY — Click "Start Scan" in the sidebar to begin</div>',
    unsafe_allow_html=True,
)

video_col, info_col = st.columns([2.1, 1])
frame_placeholder = video_col.empty()
rest_stop_placeholder = info_col.empty()

with info_col:
    st.markdown(
        """
        <div class="glass-panel" style="text-align:left;">
            <h4 style="margin-top:0; color:#e2e8f0;">🧠 Telemetry & Safety Logic</h4>
            <ul style="color:#cbd5e1; font-size:0.88rem; padding-left:18px; margin-bottom:0;">
                <li><b>Resolution:</b> Normalized 640×480 stream</li>
                <li><b>Eyes Open:</b> EAR &ge; Threshold (Active)</li>
                <li><b>Drowsy Score:</b> 0–100% real-time fatigue index</li>
                <li><b>Sustained Fatigue (&ge;3s):</b> Unlocks live nearby rest stops & emergency call</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# VIDEO PROCESSING LOOP
# ---------------------------------------------------------
if st.session_state.run_system:
    detector = load_face_landmarker()
    cap = None

    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera access error: Unable to open Webcam (index 0). Check browser/OS camera permissions or switch to 'Upload Video'.")
            st.session_state.run_system = False
    elif video_source == "Upload Video":
        if uploaded_file is None:
            st.warning("⚠️ Missing video upload: Please upload a video file in the sidebar before clicking Start Scan.")
            st.session_state.run_system = False
        else:
            with open(TEMP_VIDEO_FILE, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(TEMP_VIDEO_FILE)
            if not cap.isOpened():
                st.error("❌ Unable to decode the uploaded video file.")
                st.session_state.run_system = False

    if cap is not None and cap.isOpened():
        closed_frames = 0
        blink_count = 0
        alert_count = 0
        drowsy_score = 0
        alert_active = False
        critical_start_time = None
        last_beep_time = 0.0
        prev_time = time.time()
        rest_stops_rendered = False

        try:
            while st.session_state.run_system and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if video_source == "Upload Video":
                        status_placeholder.markdown(
                            '<div class="status-card status-idle">✅ Video scan complete — Metrics reset on next run.</div>',
                            unsafe_allow_html=True,
                        )
                    break

                if video_source == "Webcam":
                    frame = cv2.flip(frame, 1)

                # Resize frame to 640x480 for consistent speed & landmark scaling
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)

                avg_ear = 0.0
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    left_pts = np.array(
                        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX],
                        dtype=np.float32,
                    )
                    right_pts = np.array(
                        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX],
                        dtype=np.float32,
                    )

                    left_ear = compute_ear(left_pts)
                    right_ear = compute_ear(right_pts)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Visually validate left & right eye landmarks
                    eye_color = (0, 0, 255) if avg_ear < ear_threshold else (56, 189, 248)
                    cv2.polylines(frame, [left_pts.astype(np.int32)], True, eye_color, 2)
                    cv2.polylines(frame, [right_pts.astype(np.int32)], True, eye_color, 2)
                    for pt in np.vstack((left_pts, right_pts)).astype(np.int32):
                        cv2.circle(frame, tuple(pt), 2, (255, 255, 255), -1)

                    # Blink, Drowsy Score & Critical Timer state machine
                    if avg_ear < ear_threshold:
                        closed_frames += 1
                        drowsy_score = min(100, int((closed_frames / max(warning_frames, 1)) * 100))

                        if closed_frames >= warning_frames:
                            if not alert_active:
                                alert_count += 1
                                alert_active = True
                                if critical_start_time is None:
                                    critical_start_time = time.time()

                            if time.time() - last_beep_time > 1.2:
                                async_beep()
                                last_beep_time = time.time()
                    else:
                        if 2 <= closed_frames < warning_frames:
                            blink_count += 1
                        closed_frames = 0
                        alert_active = False
                        critical_start_time = None
                        drowsy_score = max(0, drowsy_score - 15)

                # FPS Calculation
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # HUD Overlay on 640x480 Frame
                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f} | Score: {drowsy_score}%",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if alert_active else (52, 211, 153),
                    2,
                )

                if alert_active:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT! PULL OVER!",
                        (14, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                # Update all 5 glass-panel metrics on every processed frame
                update_metrics(fps, avg_ear, drowsy_score, blink_count, alert_count)

                # Update Status Card Banner
                if alert_active:
                    status_placeholder.markdown(
                        '<div class="status-card status-warning">🚨 CRITICAL DROWSINESS DETECTED — PULL OVER SAFELY! 🚨</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_placeholder.markdown(
                        '<div class="status-card status-active">🟢 AI GUARDIAN ACTIVE — Driver Alert</div>',
                        unsafe_allow_html=True,
                    )

                # Trigger Rest-Stop Suggestions & Emergency Call after 3 seconds of sustained drowsiness
                sustained_3s = (
                    critical_start_time is not None
                    and (time.time() - critical_start_time) >= 3.0
                )
                if (sustained_3s or alert_count >= 2) and not rest_stops_rendered:
                    stops_html, maps_url, detected_city = get_live_rest_stops()
                    tel_uri = f"tel:{st.session_state.emergency_phone}"
                    rest_stop_placeholder.markdown(
                        f"""
                        <div class="rest-stop-card">
                            <h4 style="margin-top:0;">🗺️ Nearest Safe Rest Stops ({detected_city})</h4>
                            <p style="font-size:0.86rem; margin-bottom:8px;">
                                Sustained drowsiness (&ge;3s) detected. Pull over at a safe location:
                            </p>
                            <ul style="font-size:0.84rem; padding-left:18px; margin-bottom:10px;">
                                {stops_html}
                            </ul>
                            <a href="{maps_url}" target="_blank"
                               style="color:#fbbf24; font-weight:700; text-decoration:underline;">
                               📍 Open Live Rest Stops in Google Maps &rarr;
                            </a>
                            <a href="{tel_uri}" class="emergency-btn">
                               📞 Call Emergency Contact ({st.session_state.emergency_phone})
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    rest_stops_rendered = True

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )
        finally:
            # Release camera, clean up temp_vid.mp4, and reset metrics on Standby/Exit
            cap.release()
            if os.path.exists(TEMP_VIDEO_FILE):
                try:
                    os.remove(TEMP_VIDEO_FILE)
                except OSError:
                    pass
            st.session_state.run_system = False
            update_metrics(0.0, 0.0, 0, 0, 0)
            status_placeholder.markdown(
                '<div class="status-card status-idle">⚪ SYSTEM STANDBY — Camera released & metrics reset</div>',
                unsafe_allow_html=True,
            )