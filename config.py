# ============================================================
# config.py — The Control Room 🎛️
# ============================================================
# Ye file project ki SAARI settings ka single source of truth hai.
# Agar client bole "alarm thoda pehle baje" ya "detection strict karo",
# sirf is file mein ek number change karo — baaki sab automatically update ho jayega.
# ============================================================

# ==============================================================
# 👁️  EAR (Eye Aspect Ratio) — Neend / Drowsiness Detection
# ==============================================================

# Agar kisi bhi frame mein EAR is value se NEECHE gira, toh aankh "band" maani jayegi.
# Range: 0.0 (poori band) se 0.4 (poori khuli) tak.
# Lower karo → more sensitive (chhoti si jhapki bhi detect hogi)
# Higher karo → less sensitive (sirf gehri neend detect hogi)
EAR_THRESHOLD = 0.25

# Kitne LAGATAR frames tak aankh band rahni chahiye, tab ALARM bajega.
# Formula: Frames / FPS = Seconds
# Example: 20 frames / 30 FPS ≈ 0.67 seconds
EAR_CONSEC_FRAMES = 20

# ==============================================================
# 😮  MAR (Mouth Aspect Ratio) — Yawning / Thakan Detection
# ==============================================================

# Agar MAR is value se UPAR gaya, toh "yawning" consider kiya jayega.
MAR_THRESHOLD = 0.60

# Kitne lagatar frames tak yawning detect honi chahiye, alarm ke liye.
MAR_CONSEC_FRAMES = 15

# ==============================================================
# 🔔  Alert System — Alarm Settings
# ==============================================================

# Alarm ki frequency (Hz mein). Higher = zyada tez / penetrating sound.
ALARM_FREQUENCY_HZ = 1000

# Ek beep ki duration (seconds mein).
ALARM_DURATION_SEC = 0.8

# Alarm ki volume (0.0 se 1.0 tak). 1.0 = maximum.
ALARM_VOLUME = 0.9

# ==============================================================
# 🎨  Display / Drawing Settings
# ==============================================================

# OpenCV BGR format mein colors (NOT RGB!)
COLOR_GREEN  = (0, 255, 0)    # Sab theek hai
COLOR_RED    = (0, 0, 255)    # Alert / Drowsy!
COLOR_YELLOW = (0, 255, 255)  # Warning
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 255, 0)

# Landmark drawing settings
LANDMARK_RADIUS     = 1
LANDMARK_THICKNESS  = 1
CONTOUR_THICKNESS   = 2

# ==============================================================
# 📊  MediaPipe Face Mesh — Landmark Indices
# ==============================================================
# In numbers ko KABHI mat badlo jab tak MediaPipe update na ho.
# Ye officially MediaPipe Face Mesh ke 468-point model ke indices hain.

# Left Eye (6 key points for EAR calculation)
# Order: [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]
LEFT_EYE_LANDMARKS  = [362, 385, 387, 263, 373, 380]

# Right Eye (6 key points for EAR calculation)
# Order: [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Mouth (8 key points for MAR calculation)
# Order: [left, upper_left, top, upper_right, right, lower_right, bottom, lower_left]
MOUTH_LANDMARKS     = [61, 39, 0, 269, 291, 405, 17, 181]

# ==============================================================
# 📝  Status Display Labels
# ==============================================================
STATUS_AWAKE    = "AWAKE"
STATUS_DROWSY   = "DROWSY"
STATUS_SLEEPING = "SLEEPING!"
STATUS_YAWNING  = "YAWNING"
STATUS_NO_FACE  = "NO FACE DETECTED"
