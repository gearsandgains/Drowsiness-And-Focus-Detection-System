# 🚗 DrowseGuard Pro — Driver Drowsiness & Focus Detection System

> **Real-time AI-powered drowsiness detection using MediaPipe Face Mesh, OpenCV, and Streamlit-WebRTC.**

---

## 🎯 What It Does

DrowseGuard Pro monitors a driver's face in real-time and:
- 👁️ Detects **eye closure** (drowsiness) using **Eye Aspect Ratio (EAR)**
- 😮 Detects **yawning** using **Mouth Aspect Ratio (MAR)**
- 🚨 Triggers a **non-freezing audio alarm** via threading when danger is detected
- 📊 Shows **live metrics** (EAR, MAR, status) in a premium dashboard

---

## 🏗️ Project Architecture

```
drowsiness-detection/
├── app.py              🖥️  Streamlit UI + streamlit-webrtc integration
├── vision_engine.py    🧠  MediaPipe Face Mesh processing pipeline
├── geometry_utils.py   📐  EAR / MAR mathematical calculations
├── alert_system.py     🔔  Thread-safe audio alarm system
├── config.py           🎛️  All thresholds and settings (single source of truth)
└── requirements.txt    📦  Python dependencies
```

### Design Principles
| Concern | Solution |
|---|---|
| Zero-lag webcam streaming | `streamlit-webrtc` (WebRTC, not `cv2.VideoCapture`) |
| Non-freezing alarm | `threading.Thread` in `AlertSystem` |
| No external audio files | Sine wave generated via `numpy` + `pygame` |
| Lightweight AI | MediaPipe (CPU only, 30+ FPS on any laptop) |
| Easy client tuning | All numbers in `config.py` only |

---

## 🚀 Quick Start

### 1. Clone or download this project

```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, click **START**, and allow webcam access.

---

## ⚙️ Configuration

All tunable parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | Below this → eye considered closed |
| `EAR_CONSEC_FRAMES` | `20` | Frames of closed eyes before alarm (~0.67s @ 30fps) |
| `MAR_THRESHOLD` | `0.60` | Above this → yawning detected |
| `MAR_CONSEC_FRAMES` | `15` | Frames of yawning before alarm (~0.5s @ 30fps) |
| `ALARM_FREQUENCY_HZ` | `1000` | Alarm beep pitch in Hz |
| `ALARM_DURATION_SEC` | `0.8` | Duration of each beep |

> **Client tip:** If the client says "alarm bahut deri se bajata hai", reduce `EAR_CONSEC_FRAMES` in `config.py`. No other file needs to change.

---

## 🧠 Algorithm Details

### Eye Aspect Ratio (EAR)
Based on [Soukupova & Cech (2016)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf):

$$EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \cdot \|P_1 - P_4\|}$$

- Open eye → EAR ≈ 0.30 (constant)
- Closed eye → EAR ≈ 0.0

### Mouth Aspect Ratio (MAR)
Custom 8-point formula:

$$MAR = \frac{\|P_2 - P_8\| + \|P_3 - P_7\| + \|P_4 - P_6\|}{3 \cdot \|P_1 - P_5\|}$$

- Closed mouth → MAR ≈ 0.3
- Yawning → MAR > 0.6

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `streamlit-webrtc` | Zero-lag WebRTC webcam streaming |
| `opencv-python` | Frame processing & drawing |
| `mediapipe` | 468-point Face Mesh landmarks |
| `scipy` | Euclidean distance calculations |
| `numpy` | Array math & sound generation |
| `pygame` | Cross-platform audio playback |
| `av` | Video frame encoding/decoding for WebRTC |

---

## 🛡️ Why This Won't Freeze

| Problem | Our Solution |
|---|---|
| Streamlit + OpenCV loop → hang | `streamlit-webrtc` handles video in WebRTC thread |
| Alarm playback → video pause | `threading.Thread(daemon=True)` in `AlertSystem` |
| Heavy ML model → low FPS | MediaPipe is CPU-optimized (no GPU needed) |
| External audio file dependency | Sine wave generated with `numpy` at runtime |

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ using MediaPipe · OpenCV · Streamlit*
