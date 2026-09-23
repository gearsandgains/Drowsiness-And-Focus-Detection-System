# ============================================================
# alert_system.py — The Voice 🔔
# ============================================================
# Ye file SIRF alarm bajane ka kaam karti hai.
# CRITICAL: Alarm ko BACKGROUND THREAD mein bajaya jaata hai.
#
# Agar alarm main thread mein bajao → video feed freeze ho jaayegi.
# Threading se alarm aur video dono simultaneously run karte hain.
#
# Sound Engine: Pygame (cross-platform, reliable)
# Fallback: numpy se programmatically sine wave generate karta hai —
# koi external .mp3/.wav file chahiye hi nahi!
# ============================================================

import threading
import numpy as np
import pygame
import logging

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Thread-safe alarm system jo drowsiness ya yawning detect hone par bajta hai.

    Design Pattern: Singleton-like — ek hi instance rakho pure app mein.

    Usage:
        alert = AlertSystem()
        alert.trigger_alarm()   # Alarm start karo (non-blocking)
        alert.stop_alarm()      # Alarm band karo
        alert.cleanup()         # App band karte waqt resources free karo
    """

    def __init__(self, frequency: int = 1000, duration: float = 0.8, volume: float = 0.9):
        """
        AlertSystem initialize karta hai.

        Args:
            frequency: Beep tone frequency in Hz (default: 1000Hz)
            duration:  Ek beep ki duration in seconds (default: 0.8s)
            volume:    Volume level 0.0 to 1.0 (default: 0.9)
        """
        self.frequency = frequency
        self.duration = duration
        self.volume = volume

        # Thread safety ke liye lock — ek waqt mein sirf ek thread alarm control kare
        self._lock = threading.Lock()

        # Alarm currently chal raha hai ya nahi
        self._is_playing = False

        # Current alarm thread ka reference (agar chal raha ho)
        self._alarm_thread: threading.Thread | None = None

        # Pygame audio engine initialize karo
        self._audio_ready = self._initialize_pygame()

        # Sound buffer pehle se generate karke rakho (performance ke liye)
        self._sound: pygame.Sound | None = None
        if self._audio_ready:
            self._sound = self._generate_beep_sound()

    def _initialize_pygame(self) -> bool:
        """
        Pygame mixer ko initialize karta hai.

        Returns:
            bool: True agar initialization successful, False agar fail hua
        """
        try:
            # Agar pygame already initialized hai toh skip karo
            if not pygame.get_init():
                pygame.init()

            # Mixer settings: 44100 Hz sample rate, 16-bit audio, stereo, 512 buffer
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            logger.info("AlertSystem: Pygame audio initialized successfully.")
            return True

        except Exception as e:
            logger.warning(f"AlertSystem: Pygame init failed: {e}. Alarm disabled.")
            return False

    def _generate_beep_sound(self) -> pygame.Sound | None:
        """
        Numpy se programmatically ek sine wave beep sound generate karta hai.

        Koi external audio file ki zaroorat nahi — pure math se sound banta hai!
        Formula: wave = sin(2π * frequency * time)

        Returns:
            pygame.Sound object ya None agar error aaya
        """
        try:
            sample_rate = 44100
            num_samples = int(sample_rate * self.duration)

            # Time array banao (0 se duration tak)
            t = np.linspace(0, self.duration, num_samples, endpoint=False)

            # Sine wave generate karo aur 16-bit integer range mein scale karo
            wave = np.sin(2 * np.pi * self.frequency * t)

            # Smoothing: fade in aur fade out lagao (harsh click sound avoid karne ke liye)
            fade_samples = int(sample_rate * 0.05)  # 50ms fade
            fade_in  = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples]  *= fade_in
            wave[-fade_samples:] *= fade_out

            # 16-bit integer mein convert karo aur stereo banao
            wave_int16 = (wave * 32767 * self.volume).astype(np.int16)
            stereo_wave = np.column_stack([wave_int16, wave_int16])

            # Ensure C-contiguous array (pygame ki requirement)
            stereo_wave = np.ascontiguousarray(stereo_wave)

            sound = pygame.sndarray.make_sound(stereo_wave)
            logger.info("AlertSystem: Beep sound generated successfully.")
            return sound

        except Exception as e:
            logger.error(f"AlertSystem: Sound generation failed: {e}")
            return None

    def _play_loop(self):
        """
        Background thread mein continuously alarm bajata rehta hai.
        Jab tak self._is_playing True hai, tab tak bajata rehta hai.

        Yeh method DIRECTLY CALL NAHI KARNA — sirf trigger_alarm() se call hota hai.
        """
        while True:
            with self._lock:
                if not self._is_playing:
                    break  # Stop signal mila, loop khatam karo

            # Sound bajao (agar available hai)
            if self._sound and self._audio_ready:
                try:
                    self._sound.play()
                    pygame.time.wait(int(self.duration * 1000))  # Duration tak wait karo
                except Exception as e:
                    logger.error(f"AlertSystem: Playback error: {e}")
                    break
            else:
                # Fallback: agar pygame kaam nahi kar raha
                import time
                time.sleep(self.duration)

    def trigger_alarm(self):
        """
        Alarm start karta hai — NON-BLOCKING (video freeze nahi hogi).

        Agar alarm pehle se chal raha hai, kuch nahi hoga (double-trigger safe).
        """
        with self._lock:
            if self._is_playing:
                return  # Pehle se chal raha hai, skip karo

            self._is_playing = True

        # Naya background thread start karo
        self._alarm_thread = threading.Thread(
            target=self._play_loop,
            daemon=True,  # Main app band hone par ye thread bhi automatically band hoga
            name="AlarmThread"
        )
        self._alarm_thread.start()
        logger.info("AlertSystem: 🚨 Alarm TRIGGERED.")

    def stop_alarm(self):
        """
        Alarm band karta hai.

        Thread ko signal deta hai rukne ka, aur pygame ka current playback bhi rokta hai.
        """
        with self._lock:
            if not self._is_playing:
                return  # Pehle se band hai
            self._is_playing = False

        # Pygame ka current playback immediately rok do
        if self._audio_ready:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

        logger.info("AlertSystem: ✅ Alarm STOPPED.")

    def is_playing(self) -> bool:
        """Kya alarm currently chal raha hai?"""
        with self._lock:
            return self._is_playing

    def cleanup(self):
        """
        Saare resources release karta hai.
        App band karte waqt zaroor call karo (Streamlit ke on_exit mein).
        """
        self.stop_alarm()
        if self._audio_ready:
            try:
                pygame.mixer.quit()
                pygame.quit()
            except Exception:
                pass
        logger.info("AlertSystem: Resources cleaned up.")
