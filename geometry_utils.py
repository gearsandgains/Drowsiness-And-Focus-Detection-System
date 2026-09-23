# ============================================================
# geometry_utils.py — The Math 📐
# ============================================================
# Is file mein sirf pure mathematics hai — koi OpenCV, koi MediaPipe nahi.
# EAR aur MAR calculate karne ke functions yahan hain.
# Inhe alag rakhne se code testable aur reusable rehta hai.
#
# EAR Formula (Soukupova & Cech, 2016):
#   EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
#
# MAR Formula (Customized for 8-point mouth detection):
#   MAR = (||P2-P8|| + ||P3-P7|| + ||P4-P6||) / (3 * ||P1-P5||)
# ============================================================

import numpy as np
from scipy.spatial import distance as dist


def calculate_ear(eye_landmarks: list) -> float:
    """
    Eye Aspect Ratio (EAR) calculate karta hai.

    Jab aankh khuli hoti hai, EAR ek constant (≈0.3) rehta hai.
    Jab aankh band hoti hai, EAR 0 ke kareeb aa jaata hai.
    Yahi cheez drowsiness detect karne mein kaam aati hai.

    Args:
        eye_landmarks: 6 (x, y) points ki list, is order mein:
                       [P1_left, P2_upper_left, P3_upper_right,
                        P4_right, P5_lower_right, P6_lower_left]

    Returns:
        float: EAR value (0.0 = poori band, ~0.3+ = khuli)
    """
    # Aankh ke vertical distances (upar-neeche)
    # P2 se P6 ki vertical distance (left vertical)
    vertical_dist_1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])

    # P3 se P5 ki vertical distance (right vertical)
    vertical_dist_2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Aankh ki horizontal distance (left corner se right corner)
    horizontal_dist = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # EAR formula: vertical average / horizontal distance
    # Division by 2 isliye kyunki hum 2 vertical distances use kar rahe hain
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return ear


def calculate_mar(mouth_landmarks: list) -> float:
    """
    Mouth Aspect Ratio (MAR) calculate karta hai — yawning detect karne ke liye.

    Jab muh banda hota hai, MAR chhota hota hai (≈0.3-0.4).
    Jab muh khulta hai (yawning), MAR badh jaata hai (≈0.6+).

    Args:
        mouth_landmarks: 8 (x, y) points ki list, is order mein:
                         [P1_left, P2_upper_left, P3_top,  P4_upper_right,
                          P5_right, P6_lower_right, P7_bottom, P8_lower_left]

    Returns:
        float: MAR value (chhota = band muh, bada = khula muh/yawning)
    """
    # Teen vertical distances measure karte hain (muh ke khulne ki depth)
    # P2 (upper-left) se P8 (lower-left) ki distance
    v1 = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])

    # P3 (top center) se P7 (bottom center) ki distance
    v2 = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])

    # P4 (upper-right) se P6 (lower-right) ki distance
    v3 = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])

    # Muh ki horizontal distance (left corner se right corner)
    horizontal_dist = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])

    # MAR formula: average vertical distance / horizontal distance
    mar = (v1 + v2 + v3) / (3.0 * horizontal_dist)
    return mar


def extract_landmark_coords(face_landmarks, indices: list, img_w: int, img_h: int) -> list:
    """
    MediaPipe ke normalized (0-1) landmark coordinates ko pixel coordinates mein convert karta hai.

    MediaPipe landmarks 0 to 1 scale mein hote hain, isliye
    unhe image width/height se multiply karke pixel position milta hai.

    Args:
        face_landmarks: MediaPipe face_landmarks object (ek face ka)
        indices:        Jinhe extract karna hai unke landmark index numbers
        img_w:          Image width in pixels
        img_h:          Image height in pixels

    Returns:
        list: (x, y) pixel coordinate tuples ki list (indices ke order mein)
    """
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        # Normalized coordinates (0-1) ko actual pixel position mein convert karo
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        coords.append((x, y))
    return coords
