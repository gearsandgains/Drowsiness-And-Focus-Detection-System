# ============================================================
# app.py — The Face / UI 🖥️
# ============================================================
# Ye project ka main entry point hai.
# Run karo: streamlit run app.py
#
# Features:
#   ✅ Premium dark-mode UI (custom CSS)
#   ✅ streamlit-webrtc (zero-lag webcam streaming)
#   ✅ Real-time EAR, MAR, status metrics sidebar mein
#   ✅ Alarm system threading ke saath (video freeze nahi hogi)
#   ✅ Session-based stats tracking
#
# Architecture Note:
#   streamlit-webrtc alag thread mein chalta hai.
#   VideoProcessor class ke andar engine aur alert system hain.
#   UI aur VideoProcessor ke beech communication ke liye
#   threading.Lock() + shared state use kiya gaya hai.
# ============================================================

import streamlit as st
import av
import cv2
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import config
from vision_engine import VisionEngine, ProcessedResult
from alert_system import AlertSystem

# ==============================================================
# 🎨  Page Configuration — Sabse pehle call karna ZAROORI hai
# ==============================================================
st.set_page_config(
    page_title="DrowseGuard Pro — Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# 💅  Custom CSS — Premium Dark Mode UI
# ==============================================================
CUSTOM_CSS = """
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ---- Main App Background ---- */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(100, 100, 200, 0.2);
    }

    /* ---- Metric Cards ---- */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        backdrop-filter: blur(10px);
    }

    [data-testid="metric-container"] label {
        color: #a0a0c0 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ---- Status Badge ---- */
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        margin: 8px 0;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .status-awake    { background: rgba(0,200,80,0.15);  border: 2px solid #00c850; color: #00c850; }
    .status-drowsy   { background: rgba(255,200,0,0.15); border: 2px solid #ffc800; color: #ffc800; }
    .status-sleeping { background: rgba(255,50,50,0.20); border: 2px solid #ff3232; color: #ff3232; animation: alarm-pulse 0.5s ease-in-out infinite; }
    .status-yawning  { background: rgba(255,150,0,0.15); border: 2px solid #ff9600; color: #ff9600; }
    .status-noface   { background: rgba(120,120,120,0.15); border: 2px solid #888; color: #888; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.7; }
    }
    @keyframes alarm-pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255,50,50,0.8); }
        50%       { box-shadow: 0 0 40px rgba(255,50,50,0.4); }
    }

    /* ---- Section Headers ---- */
    .section-header {
        color: #7070ff;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(112, 112, 255, 0.3);
    }

    /* ---- Info Cards ---- */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #b0b0c8;
    }

    /* ---- Threshold Progress Bars ---- */
    .stProgress > div > div > div { border-radius: 10px; }

    /* ---- Divider ---- */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================
# 🔧  WebRTC Configuration — STUN/TURN Servers
# ==============================================================
# WebRTC ko browser aur server ke beech connection establish karne ke liye
# STUN server chahiye. Google ka public STUN server free mein use karo.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


# ==============================================================
# 🎥  Video Processor — The Heart of streamlit-webrtc
# ==============================================================
class DrowsinessVideoProcessor(VideoProcessorBase):
    """
    streamlit-webrtc ka VideoProcessor — har webcam frame yahan process hota hai.

    Ye class BACKGROUND THREAD mein chalta hai (Streamlit ke main thread se alag).
    Isliye saari shared state (metrics, counters) thread-safe tarike se access karni hai.

    Design:
      - VisionEngine: Frame processing (MediaPipe + OpenCV)
      - AlertSystem: Alarm management (threading)
      - threading.Lock(): UI aur video thread ke beech safe data sharing
    """

    def __init__(self):
        # Core processing engine
        self.engine = VisionEngine()

        # Alarm system (threading ke saath)
        self.alert = AlertSystem(
            frequency=config.ALARM_FREQUENCY_HZ,
            duration=config.ALARM_DURATION_SEC,
            volume=config.ALARM_VOLUME
        )

        # Thread-safe shared state — UI yahan se metrics padhega
        self._lock = threading.Lock()
        self._last_result = ProcessedResult()

        # Session statistics
        self._session_start_time    = time.time()
        self._total_drowsy_frames   = 0
        self._total_yawn_events     = 0
        self._alarm_triggered_count = 0
        self._prev_alarm_state      = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        streamlit-webrtc is function ko har frame ke liye call karta hai.

        1. av.VideoFrame → numpy BGR array
        2. VisionEngine se process karo
        3. Alarm logic run karo
        4. numpy BGR array → av.VideoFrame (return karo)
        """
        # av.VideoFrame ko OpenCV BGR numpy array mein convert karo
        img = frame.to_ndarray(format="bgr24")

        # Vision engine se frame process karo
        result = self.engine.process_frame(img)

        # Alarm logic: agar should_alarm hai toh trigger karo, warna band karo
        if result.should_alarm:
            self.alert.trigger_alarm()
            # Naya alarm event track karo
            if not self._prev_alarm_state:
                self._alarm_triggered_count += 1
        else:
            self.alert.stop_alarm()

        self._prev_alarm_state = result.should_alarm

        # Session stats update karo (thread-safe)
        with self._lock:
            self._last_result = result
            if result.is_drowsy:
                self._total_drowsy_frames += 1
            if result.is_yawning:
                self._total_yawn_events += 1

        # Annotated frame ko wapas av.VideoFrame format mein convert karo
        return av.VideoFrame.from_ndarray(result.annotated_frame, format="bgr24")

    def get_latest_result(self) -> ProcessedResult:
        """Thread-safe method: UI ko latest result deta hai."""
        with self._lock:
            return self._last_result

    def get_session_stats(self) -> dict:
        """Session statistics return karta hai."""
        elapsed = time.time() - self._session_start_time
        with self._lock:
            return {
                "elapsed_seconds": elapsed,
                "total_drowsy_frames": self._total_drowsy_frames,
                "alarm_count": self._alarm_triggered_count,
            }

    def __del__(self):
        """Cleanup jab processor destroy ho."""
        if hasattr(self, 'alert'):
            self.alert.cleanup()


# ==============================================================
# 🖥️  MAIN UI LAYOUT
# ==============================================================

# --- Header ---
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 style="font-size:2.8rem; font-weight:700; margin:0;
               background: linear-gradient(135deg, #667eea, #a78bfa, #ec4899);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🚗 DrowseGuard Pro
    </h1>
    <p style="color:#8080a0; font-size:1.05rem; margin-top:6px; letter-spacing:0.05em;">
        Real-Time Driver Drowsiness & Focus Detection System
    </p>
    <p style="color:#5050a0; font-size:0.8rem;">
        Powered by MediaPipe Face Mesh · OpenCV · Streamlit-WebRTC
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <span style="font-size:3rem;">🛡️</span>
        <h3 style="color:#a78bfa; margin:4px 0 0 0; font-weight:700;">DrowseGuard</h3>
        <p style="color:#5050a0; font-size:0.75rem; margin:0;">Safety Analytics Panel</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Live Metrics Section (placeholder, updated via st.empty() later) ---
    st.markdown('<p class="section-header">📊 Live Metrics</p>', unsafe_allow_html=True)

    # Metric placeholders — in-place update karne ke liye
    ear_metric_placeholder   = st.empty()
    mar_metric_placeholder   = st.empty()
    status_badge_placeholder = st.empty()

    st.markdown("---")

    # --- Thresholds Section ---
    st.markdown('<p class="section-header">⚙️ Detection Thresholds</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <b>👁️ EAR Threshold:</b> {config.EAR_THRESHOLD}<br>
        <b>😮 MAR Threshold:</b> {config.MAR_THRESHOLD}<br>
        <b>⏱️ Drowsy Frames:</b> {config.EAR_CONSEC_FRAMES} ({config.EAR_CONSEC_FRAMES/30:.1f}s @ 30fps)<br>
        <b>⏱️ Yawn Frames:</b> {config.MAR_CONSEC_FRAMES} ({config.MAR_CONSEC_FRAMES/30:.1f}s @ 30fps)
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#5050a0; font-size:0.72rem;">Modify in <code>config.py</code></p>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Session Stats Placeholder ---
    st.markdown('<p class="section-header">📈 Session Statistics</p>', unsafe_allow_html=True)
    session_stats_placeholder = st.empty()

    st.markdown("---")

    # --- How It Works ---
    st.markdown('<p class="section-header">ℹ️ How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <b>EAR</b> (Eye Aspect Ratio) measures how open your eyes are.
        When EAR drops below the threshold for N consecutive frames → <span style="color:#ff3232">DROWSY!</span><br><br>
        <b>MAR</b> (Mouth Aspect Ratio) measures mouth openness.
        A large MAR sustained for N frames → <span style="color:#ff9600">YAWNING!</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# MAIN CONTENT — Webcam Feed + Info Cards
# ==============================================================
col_left, col_right = st.columns([2.2, 1], gap="large")

with col_left:
    st.markdown('<p class="section-header">📹 Live Camera Feed</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card" style="margin-bottom:12px;">
        🟢 <b>Click "START" below to activate your webcam.</b>
        The AI will begin monitoring in real-time. Ensure your face is clearly visible and well-lit.
        Alarm sounds when drowsiness is detected for more than
        <b>{:.1f} seconds</b>.
    </div>
    """.format(config.EAR_CONSEC_FRAMES / 30), unsafe_allow_html=True)

    # ==============================================================
    # 🎥  streamlit-webrtc — The Magic Component
    # ==============================================================
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=DrowsinessVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width":  {"ideal": 1280, "max": 1920},
                "height": {"ideal": 720,  "max": 1080},
                "frameRate": {"ideal": 30, "max": 60},
            },
            "audio": False,  # Audio chahiye nahi (alarm apna sound bajaega)
        },
        async_processing=True,  # Background thread mein processing
        video_html_attrs={
            "style": "width:100%; border-radius:16px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);"
        }
    )

with col_right:
    st.markdown('<p class="section-header">🎯 Detection Guide</p>', unsafe_allow_html=True)

    # Status legend
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:10px;">
        <div class="info-card">
            <span style="color:#00c850; font-weight:700;">🟢 AWAKE</span><br>
            <small>Driver fully alert. EAR above threshold, no yawning.</small>
        </div>
        <div class="info-card">
            <span style="color:#ffc800; font-weight:700;">🟡 DROWSY</span><br>
            <small>Eyes partially closed for a short duration. Warning state.</small>
        </div>
        <div class="info-card">
            <span style="color:#ff3232; font-weight:700;">🔴 SLEEPING!</span><br>
            <small>Eyes closed for {:.1f}+ seconds. ALARM triggered immediately!</small>
        </div>
        <div class="info-card">
            <span style="color:#ff9600; font-weight:700;">🟠 YAWNING</span><br>
            <small>Wide mouth opening detected for {:.1f}+ seconds.</small>
        </div>
    </div>
    """.format(config.EAR_CONSEC_FRAMES / 30, config.MAR_CONSEC_FRAMES / 30),
    unsafe_allow_html=True)

    st.markdown("---")

    # Technology stack
    st.markdown('<p class="section-header">⚡ Technology Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        🧠 <b>MediaPipe Face Mesh</b><br>
        <small style="color:#7070a0">468-point 3D landmarks, CPU-optimized</small><br><br>
        📐 <b>EAR / MAR Algorithms</b><br>
        <small style="color:#7070a0">Soukupova & Cech (2016) — real-time geometry</small><br><br>
        🎥 <b>streamlit-webrtc</b><br>
        <small style="color:#7070a0">WebRTC — zero-lag browser streaming</small><br><br>
        🔔 <b>Thread-safe Alarm</b><br>
        <small style="color:#7070a0">Python threading — video never freezes</small>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# REAL-TIME METRICS UPDATE LOOP
# ==============================================================
# streamlit-webrtc chal raha hai? Toh sidebar metrics update karo.
if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    processor: DrowsinessVideoProcessor = webrtc_ctx.video_processor

    # Metrics ko continuously update karo (1 FPS pe refresh)
    while webrtc_ctx.state.playing:
        try:
            result      = processor.get_latest_result()
            stats       = processor.get_session_stats()

            # --- EAR Metric ---
            ear_metric_placeholder.metric(
                label="👁️ Eye Aspect Ratio (EAR)",
                value=f"{result.avg_ear:.3f}",
                delta=f"Threshold: {config.EAR_THRESHOLD}",
                delta_color="off"
            )

            # --- MAR Metric ---
            mar_metric_placeholder.metric(
                label="😮 Mouth Aspect Ratio (MAR)",
                value=f"{result.mar:.3f}",
                delta=f"Threshold: {config.MAR_THRESHOLD}",
                delta_color="off"
            )

            # --- Status Badge ---
            status_lower = result.status.lower().replace("!", "").strip()
            if "sleeping" in status_lower:
                badge_class = "status-sleeping"
            elif "drowsy" in status_lower:
                badge_class = "status-drowsy"
            elif "yawning" in status_lower:
                badge_class = "status-yawning"
            elif "awake" in status_lower:
                badge_class = "status-awake"
            else:
                badge_class = "status-noface"

            status_badge_placeholder.markdown(
                f'<div class="status-badge {badge_class}">{result.status}</div>',
                unsafe_allow_html=True
            )

            # --- Session Stats ---
            elapsed_min = stats["elapsed_seconds"] / 60
            session_stats_placeholder.markdown(f"""
            <div class="info-card">
                ⏱️ <b>Session:</b> {elapsed_min:.1f} min<br>
                😴 <b>Drowsy Frames:</b> {stats["total_drowsy_frames"]}<br>
                🚨 <b>Alarms:</b> {stats["alarm_count"]}
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            # Agar processor abhi ready nahi hai, skip karo
            pass

        time.sleep(0.5)  # 2 FPS pe update (CPU save karne ke liye)

else:
    # Webcam band hai — default values dikhao
    ear_metric_placeholder.metric("👁️ Eye Aspect Ratio (EAR)", "—")
    mar_metric_placeholder.metric("😮 Mouth Aspect Ratio (MAR)", "—")
    status_badge_placeholder.markdown(
        '<div class="status-badge status-noface">📷 Camera Offline</div>',
        unsafe_allow_html=True
    )
    session_stats_placeholder.markdown("""
    <div class="info-card" style="color:#5050a0;">
        Click <b>START</b> to begin monitoring.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# FOOTER
# ==============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a3a5c; font-size:0.78rem; padding: 10px 0 20px 0;">
    <b>DrowseGuard Pro</b> · Built with MediaPipe, OpenCV & Streamlit-WebRTC<br>
    <span style="color:#2a2a4c;">For demonstration and research purposes. Not a substitute for proper rest.</span>
</div>
""", unsafe_allow_html=True)
