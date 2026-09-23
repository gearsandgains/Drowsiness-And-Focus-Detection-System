# ============================================================
# vision_engine.py — The Brain 🧠
# ============================================================
# Is file ka kaam ek hi hai: webcam ka ek frame lo, process karo,
# aur annotated frame + saari metrics wapas bhejo.
#
# Pipeline:
#   Raw Frame (BGR)
#     → MediaPipe Face Mesh (468 landmarks detect karo)
#     → geometry_utils se EAR/MAR calculate karo
#     → State update karo (drowsy counter, yawn counter)
#     → Frame par annotations draw karo
#     → Annotated Frame + ProcessedResult return karo
#
# Key Technology: MediaPipe Face Mesh
#   - CPU par bhi 30+ FPS deta hai
#   - No GPU required
#   - 468 precise 3D facial landmarks
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config
from geometry_utils import calculate_ear, calculate_mar, extract_landmark_coords


@dataclass
class ProcessedResult:
    """
    Vision Engine ka output — ek frame process karne ke baad ye data milta hai.
    Streamlit UI ko exactly ye chahiye hota hai display karne ke liye.
    """
    # Annotated frame (landmarks, text, colored boxes ke saath)
    annotated_frame: np.ndarray = field(default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))

    # Current metrics
    left_ear: float   = 0.0   # Left eye ka EAR
    right_ear: float  = 0.0   # Right eye ka EAR
    avg_ear: float    = 0.0   # Average EAR (dono aankhon ka)
    mar: float        = 0.0   # Mouth Aspect Ratio (yawning)

    # Counters
    drowsy_frame_count: int = 0  # Kitne frames se aankh band hai
    yawn_frame_count: int   = 0  # Kitne frames se yawning hai

    # Status flags
    is_drowsy: bool      = False  # Kya driver drowsy hai?
    is_yawning: bool     = False  # Kya driver yawn kar raha hai?
    should_alarm: bool   = False  # Kya alarm bajna chahiye?
    face_detected: bool  = False  # Kya koi chehra dikh raha hai?

    # Display status string
    status: str = config.STATUS_NO_FACE


class VisionEngine:
    """
    MediaPipe Face Mesh aur OpenCV ka use karke drowsiness detect karta hai.

    Is class ka ek hi instance create karo (Streamlit session state mein rakho)
    aur har frame ke liye process_frame() call karo.

    Thread Safety Note: streamlit-webrtc alag thread mein run karta hai.
    Is engine ke state variables ko VideoProcessor class mein manage karo.
    """

    def __init__(self):
        """
        MediaPipe Face Mesh initialize karta hai.
        Ye ek baar hota hai — initialization expensive hai, isliye reuse karo.
        """
        # MediaPipe Face Mesh setup
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing   = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # Face Mesh model initialize karo
        # refine_landmarks=True: Irises bhi detect karta hai (better accuracy)
        # min_detection_confidence: Yahan se confident hone par hi face count karo
        # min_tracking_confidence: Tracking kitni confident honi chahiye
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,           # Sirf ek driver ka chehra track karo
            refine_landmarks=True,     # Iris landmarks bhi include karo
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State counters (har frame mein update hote hain)
        self._drowsy_frame_count = 0
        self._yawn_frame_count   = 0

        print("✅ VisionEngine: MediaPipe Face Mesh initialized.")

    def process_frame(self, frame: np.ndarray) -> ProcessedResult:
        """
        Ek BGR frame process karta hai aur ProcessedResult return karta hai.

        Yahi function streamlit-webrtc ke VideoProcessor.recv() se call hota hai.
        Har frame par ye pipeline chalti hai:
        1. BGR → RGB convert (MediaPipe RGB chahta hai)
        2. Face Mesh run karo
        3. Landmarks se EAR/MAR calculate karo
        4. State counters update karo
        5. Frame annotate karo

        Args:
            frame: OpenCV BGR format mein numpy array (H, W, 3)

        Returns:
            ProcessedResult with annotated frame and all metrics
        """
        result = ProcessedResult()
        img_h, img_w = frame.shape[:2]

        # MediaPipe ko RGB chahiye, OpenCV BGR deta hai — convert karo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Optimization: MediaPipe ke dौरान frame ko read-only banao (speed up)
        rgb_frame.flags.writeable = False
        detection_result = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Working copy banao (original frame ke upar draw karenge)
        annotated = frame.copy()

        # Agar koi chehra detect nahi hua
        if not detection_result.multi_face_landmarks:
            result.face_detected = False
            result.status = config.STATUS_NO_FACE
            self._drowsy_frame_count = 0  # Counter reset karo (face lost)
            self._yawn_frame_count   = 0

            # Frame par message likho
            self._draw_no_face_message(annotated)
            result.annotated_frame = annotated
            return result

        # Pehla (aur single) face ka landmarks lo
        face_landmarks = detection_result.multi_face_landmarks[0]
        result.face_detected = True

        # --- Landmarks ko pixel coordinates mein convert karo ---
        left_eye_pts  = extract_landmark_coords(face_landmarks, config.LEFT_EYE_LANDMARKS,  img_w, img_h)
        right_eye_pts = extract_landmark_coords(face_landmarks, config.RIGHT_EYE_LANDMARKS, img_w, img_h)
        mouth_pts     = extract_landmark_coords(face_landmarks, config.MOUTH_LANDMARKS,      img_w, img_h)

        # --- EAR aur MAR calculate karo ---
        left_ear  = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = calculate_mar(mouth_pts)

        result.left_ear  = left_ear
        result.right_ear = right_ear
        result.avg_ear   = avg_ear
        result.mar       = mar

        # --- Drowsiness State Machine ---
        # EAR threshold se neeche hai?
        if avg_ear < config.EAR_THRESHOLD:
            self._drowsy_frame_count += 1
        else:
            self._drowsy_frame_count = 0  # Aankhein khul gayin, reset karo

        # Yawning State Machine
        if mar > config.MAR_THRESHOLD:
            self._yawn_frame_count += 1
        else:
            self._yawn_frame_count = 0

        result.drowsy_frame_count = self._drowsy_frame_count
        result.yawn_frame_count   = self._yawn_frame_count

        # --- Status aur Alarm Decide karo ---
        is_sleeping = self._drowsy_frame_count >= config.EAR_CONSEC_FRAMES
        is_yawning  = self._yawn_frame_count   >= config.MAR_CONSEC_FRAMES
        is_drowsy   = (self._drowsy_frame_count > 3) and not is_sleeping  # Warning state

        result.is_drowsy    = is_drowsy or is_sleeping
        result.is_yawning   = is_yawning
        result.should_alarm = is_sleeping or is_yawning  # Alarm kab bajao

        # Status string assign karo (priority order mein)
        if is_sleeping:
            result.status = config.STATUS_SLEEPING
        elif is_drowsy:
            result.status = config.STATUS_DROWSY
        elif is_yawning:
            result.status = config.STATUS_YAWNING
        else:
            result.status = config.STATUS_AWAKE

        # --- Frame par Annotations Draw karo ---
        self._annotate_frame(annotated, face_landmarks, img_w, img_h,
                             left_eye_pts, right_eye_pts, mouth_pts,
                             avg_ear, mar, result.status, result.should_alarm)

        result.annotated_frame = annotated
        return result

    def _annotate_frame(self, frame, face_landmarks, img_w, img_h,
                        left_eye_pts, right_eye_pts, mouth_pts,
                        avg_ear, mar, status, should_alarm):
        """
        Frame par visual annotations draw karta hai — eye/mouth contours, metrics text.

        Args:
            frame: OpenCV frame (in-place modify hoga)
            ... baaki sab result data
        """
        # Status ke hisaab se overlay color decide karo
        if should_alarm:
            overlay_color = config.COLOR_RED
        elif status == config.STATUS_DROWSY:
            overlay_color = config.COLOR_YELLOW
        else:
            overlay_color = config.COLOR_GREEN

        # === Eye Contours Draw Karo ===
        self._draw_eye_contour(frame, left_eye_pts,  overlay_color)
        self._draw_eye_contour(frame, right_eye_pts, overlay_color)

        # === Mouth Contour Draw Karo ===
        self._draw_mouth_contour(frame, mouth_pts, overlay_color)

        # === Metric Text Overlay ===
        # EAR value top-left mein
        cv2.putText(frame, f"EAR: {avg_ear:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.COLOR_WHITE, 2, cv2.LINE_AA)

        # MAR value EAR ke neeche
        cv2.putText(frame, f"MAR: {mar:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.COLOR_WHITE, 2, cv2.LINE_AA)

        # Status badge — frame ke bottom mein bold text
        (text_w, text_h), _ = cv2.getTextSize(
            status, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
        )
        # Status ke peeche filled rectangle background
        cv2.rectangle(frame,
                      (10, frame.shape[0] - text_h - 20),
                      (text_w + 20, frame.shape[0] - 5),
                      overlay_color, -1)  # -1 = filled
        cv2.putText(frame, status,
                    (15, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (0, 0, 0), 2, cv2.LINE_AA)  # Black text on colored bg

        # Alarm chal raha hai? Pulsating red border add karo
        if should_alarm:
            border_thickness = 8
            cv2.rectangle(frame,
                          (0, 0),
                          (frame.shape[1] - 1, frame.shape[0] - 1),
                          config.COLOR_RED, border_thickness)

    def _draw_eye_contour(self, frame, eye_pts: list, color: tuple):
        """Eye ke landmark points ko lines se connect karke contour banata hai."""
        pts = np.array(eye_pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color,
                      thickness=config.CONTOUR_THICKNESS)

        # Har landmark par ek chhota dot
        for pt in eye_pts:
            cv2.circle(frame, pt, config.LANDMARK_RADIUS, color,
                       config.LANDMARK_THICKNESS)

    def _draw_mouth_contour(self, frame, mouth_pts: list, color: tuple):
        """Mouth ke landmark points ko connect karke outline banata hai."""
        pts = np.array(mouth_pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color,
                      thickness=config.CONTOUR_THICKNESS)

    def _draw_no_face_message(self, frame):
        """Jab koi chehra detect na ho, tab frame par message dikhata hai."""
        msg = "No Face Detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cx = (frame.shape[1] - tw) // 2
        cy = (frame.shape[0] + th) // 2
        cv2.putText(frame, msg, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    config.COLOR_YELLOW, 2, cv2.LINE_AA)

    def reset_counters(self):
        """State counters reset karta hai (session restart hone par call karo)."""
        self._drowsy_frame_count = 0
        self._yawn_frame_count   = 0

    def __del__(self):
        """Cleanup: MediaPipe resources release karo."""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()
