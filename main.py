import sys
import time
import math
import threading

import cv2
import numpy as np
import mediapipe as mp

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMainWindow,
    QSlider,
    QGroupBox,
    QFormLayout,
    QPushButton,
)


# -------- Video thread: capture and hand detection --------
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        self.mp_hands = mp.solutions.hands
        # tune confidences a bit for robustness
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.last_click_time = 0
        # runtime params
        self.smoothing = 0.5
        self.pinch_threshold = 0.05
        self.click_cooldown = 0.4
        self.last_index = None
        self.last_thumb = None
        self._bad_frame_count = 0
        # process MediaPipe only every N frames (helps on slower machines)
        self.process_every_n = 3  # set to 3 to process every 3rd frame
        self._frame_counter = 0

    def run(self):
        """Capture loop: read frames, run MediaPipe, emit frames and latest landmarks.
        This loop is defensive: catches per-frame exceptions and continues so the camera
        feed doesn't silently stop when processing fails on certain frames.
        """
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True

        try:
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        # increment bad read counter; try reopening if persistent
                        self._bad_frame_count += 1
                        if self._bad_frame_count > 30:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            time.sleep(0.2)
                            try:
                                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                            except Exception:
                                pass
                            self._bad_frame_count = 0
                        time.sleep(0.05)
                        continue
                    else:
                        self._bad_frame_count = 0

                    frame = cv2.flip(frame, 1)  # mirror
                    # get frame size early
                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # downscale a bit for processing to reduce CPU and increase robustness
                    try:
                        scale = 0.6
                        small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
                    except Exception:
                        small = rgb

                    # run MediaPipe, but throttle processing to every Nth frame to reduce CPU
                    results = None
                    try:
                        if (self._frame_counter % max(1, self.process_every_n)) == 0:
                            results = self.hands.process(small)
                    except Exception as e:
                        print(f"MediaPipe process error: {e}")
                        results = None

                    index_coords = None
                    thumb_coords = None

                    if results and results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        # landmark 8 = index fingertip, 4 = thumb_tip
                        lm_index = hand.landmark[8]
                        lm_thumb = hand.landmark[4]

                        index_coords = (lm_index.x, lm_index.y)
                        thumb_coords = (lm_thumb.x, lm_thumb.y)

                        # draw landmarks for feedback
                        for lm in hand.landmark:
                            try:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                            except Exception:
                                # if any landmark mapping fails, skip that landmark
                                continue

                        # draw line between thumb and index
                        try:
                            x1, y1 = int(lm_index.x * w), int(lm_index.y * h)
                            x2, y2 = int(lm_thumb.x * w), int(lm_thumb.y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 50), 2)
                        except Exception:
                            pass

                    # Emit frame and hand coords (no on-frame text overlay)
                    display = frame.copy()
                    self.last_index = index_coords
                    self.last_thumb = thumb_coords
                    # increment frame counter so throttling works evenly
                    self._frame_counter += 1
                    self.frame_ready.emit(display)

                except Exception as e:
                    # catch any per-frame exception so the camera loop continues
                    msg = str(e)
                    print(f"Frame loop error: {msg}")
                    time.sleep(0.05)
                    continue
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.wait()


# -------- Main Window / UI --------
class StylishWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.animation_phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(40)  # 25 fps
        # multi-cursor state (we'll render several cursors that move together)
        self.multi_count = 5
        # positions for each cursor (normalized)
        self.cursor_pos = [(0.5, 0.5) for _ in range(self.multi_count)]
        # per-cursor lerp factors (higher = faster follow)
        self.cursor_lerp = [0.75, 0.62, 0.5, 0.38, 0.28]
        # target (the main reference point set by the app)
        self.target_x = 0.5
        self.target_y = 0.5
        self.cursor_clicked = False
        # per-cursor trails
        self.trails = [[] for _ in range(self.multi_count)]

    def step(self):
        self.animation_phase += 0.06
        # advance each cursor towards the shared target
        for i in range(self.multi_count):
            cx, cy = self.cursor_pos[i]
            lerp = self.cursor_lerp[i]
            # move fraction towards target; clamp
            nx = cx + (self.target_x - cx) * lerp
            ny = cy + (self.target_y - cy) * lerp
            self.cursor_pos[i] = (nx, ny)
            # update trail for this cursor
            t = self.trails[i]
            t.insert(0, (nx, ny))
            if len(t) > 10:
                t.pop()
        self.update()

    def set_cursor(self, nx: float, ny: float, clicked: bool):
        # set the shared target for the group of cursors
        self.target_x = max(0.0, min(1.0, nx))
        self.target_y = max(0.0, min(1.0, ny))
        self.cursor_clicked = bool(clicked)
        # also seed the first cursor immediately so visuals appear responsive
        if self.cursor_pos:
            self.cursor_pos[0] = (self.target_x, self.target_y)
            # add to first cursor trail too
            self.trails[0].insert(0, (self.target_x, self.target_y))
            if len(self.trails[0]) > 10:
                self.trails[0].pop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 12))

        w = self.width()
        h = self.height()

        # Draw stacked colorful rectangles like the sample
        colors = [QColor(102, 178, 255), QColor(255, 153, 204), QColor(102, 255, 178), QColor(200, 180, 255)]
        n = 5
        base_margin = 24
        for i in range(n):
            phase = math.sin(self.animation_phase + i * 0.6)
            margin = base_margin + int(8 * i + 12 * phase)
            rect = (margin, margin + i * 4, w - 2 * margin, h - 2 * margin - i * 4)
            col = colors[i % len(colors)]
            pen_width = 3
            painter.setPen(col)
            painter.drawRect(rect[0], rect[1], rect[2], rect[3])

        # Draw multiple cursors and their trails
        size = min(w, h) // 10
        # color palette for the cursors (B,G,R order in QColor)
        palette = [QColor(102, 178, 255), QColor(255, 153, 204), QColor(102, 255, 178), QColor(200, 180, 255), QColor(180, 255, 220)]
        painter.setPen(Qt.NoPen)

        for j in range(self.multi_count):
            # draw trail for cursor j
            trail = self.trails[j]
            for k, (tx, ty) in enumerate(trail):
                px = int(tx * w)
                py = int(ty * h)
                alpha = max(20, 200 - k * 22)
                radius = max(3, int(size * (0.6 - k * 0.04)))
                col = palette[j % len(palette)]
                col.setAlpha(alpha)
                painter.setBrush(col)
                painter.drawEllipse(px - radius // 2, py - radius // 2, radius, radius)

        # draw each cursor as a triangular pointer using its color
        for j in range(self.multi_count):
            pxn, pyn = self.cursor_pos[j]
            cx = int(pxn * w)
            cy = int(pyn * h)
            col = palette[j % len(palette)]
            # slightly vary alpha per cursor for depth
            alpha = 220 - j * 26
            col.setAlpha(alpha)
            painter.setBrush(col)
            # size scale for layering
            scale = 1.0 - 0.06 * j
            p1 = QPointF(cx - int(size * scale), cy - int(size * scale / 2))
            p2 = QPointF(cx - int(size * scale / 3), cy + int(size * scale / 10))
            p3 = QPointF(cx - int(size * scale / 2), cy + int(size * scale / 2))
            painter.drawPolygon(p1, p2, p3)

        # click ripple around group (drawn centered at average position)
        if self.cursor_clicked:
            avg_x = sum(p[0] for p in self.cursor_pos) / len(self.cursor_pos)
            avg_y = sum(p[1] for p in self.cursor_pos) / len(self.cursor_pos)
            acx = int(avg_x * w)
            acy = int(avg_y * h)
            r = int(size * 2.6 * (0.6 + 0.4 * math.sin(self.animation_phase * 6)))
            pen = QColor(255, 215, 120, 160)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(acx - r // 2, acy - r // 2, r, r)

        painter.end()


# small alias for QPointF since not imported earlier
from PyQt5.QtCore import QPointF


class MouseWindow(QMainWindow):
    """Separate window that hosts the stylish virtual-mouse view."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Mouse")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.stylish = StylishWidget()
        self.stylish.setMinimumSize(360, 360)
        layout.addWidget(self.stylish)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand -> Mouse — Stylish Controller")
        self.setGeometry(120, 120, 900, 700)
        self.setStyleSheet("background-color: #0d0d0d; color: white;")

        central = QWidget()
        self.setCentralWidget(central)

        self.vbox = QVBoxLayout(central)
        self.vbox.setContentsMargins(12, 12, 12, 12)
        self.vbox.setSpacing(10)

        # Top: webcam feed (larger)
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setFixedHeight(640)
        self.video_label.setStyleSheet("background-color: black; border-radius:6px;")
        self.vbox.addWidget(self.video_label)

        # Thread for video (start early so settings can read defaults)
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.on_frame)
        self.video_thread.start()

        # virtual cursor state (normalized coordinates 0..1 inside stylish widget)
        self.virtual_x = 0.5
        self.virtual_y = 0.5
        self.virtual_click = False
        self._click_viz_until = 0.0

        # NOTE: the stylish "mouse" view will be shown in a separate window
        # Create and show the floating mouse window (so the mouse view can be large)
        self.mouse_window = MouseWindow()
        # Make the mouse window reasonably sized and show it
        self.mouse_window.resize(640, 640)
        self.mouse_window.show()

        # Timer to poll hand coords and move mouse
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_hand_and_update_virtual_cursor)
        self.poll_timer.start(16)  # ~60Hz

        # click state
        self.clicking = False
        self.last_click_time = 0

    def closeEvent(self, event):
        self.video_thread.stop()
        try:
            self.mouse_window.close()
        except Exception:
            pass
        event.accept()

    def on_frame(self, frame: np.ndarray):
        # convert frame (BGR) to QImage and show in webcam label
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qimg.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def poll_hand_and_update_virtual_cursor(self):
        # read last known landmarks from the video thread
        idx = getattr(self.video_thread, 'last_index', None)
        th = getattr(self.video_thread, 'last_thumb', None)
        if idx is None:
            return

        # idx and th are normalized (x, y) in camera coords
        nx, ny = idx
        # clamp
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # smoothing - interpolate from previous virtual coords
        s = self.video_thread.smoothing
        vx = self.virtual_x
        vy = self.virtual_y
        nx_s = vx + (nx - vx) * (1 - s)
        ny_s = vy + (ny - vy) * (1 - s)

        self.virtual_x = nx_s
        self.virtual_y = ny_s

        # detect pinch (visual click)
        if th is not None:
            dx = nx - th[0]
            dy = ny - th[1]
            dist = math.hypot(dx, dy)
            if dist < self.video_thread.pinch_threshold:
                now = time.time()
                if now - self.last_click_time > self.video_thread.click_cooldown:
                    # set visual click state
                    self.virtual_click = True
                    self._click_viz_until = now + 0.22
                    self.last_click_time = now

        # clear click visual when time expired
        if self.virtual_click and time.time() > self._click_viz_until:
            self.virtual_click = False

        # update stylish widget in the separate mouse window to show virtual cursor
        try:
            if hasattr(self, 'mouse_window') and hasattr(self.mouse_window, 'stylish'):
                self.mouse_window.stylish.set_cursor(self.virtual_x, self.virtual_y, self.virtual_click)
            elif hasattr(self, 'stylish'):
                # fallback if stylish exists locally
                self.stylish.set_cursor(self.virtual_x, self.virtual_y, self.virtual_click)
        except Exception:
            # suppress errors to avoid stopping the poll loop
            pass

    # --------- Settings callbacks ---------
    def on_smoothing_changed(self, value: int):
        # slider is 0..90 -> smoothing 0.0 .. 0.9
        s = max(0.0, min(0.9, value / 100.0))
        self.video_thread.smoothing = s
        self.smooth_label.setText(f"{s:.2f}")

    def on_pinch_changed(self, value: int):
        # slider 2..20 -> threshold 0.02 .. 0.20
        t = max(0.01, value / 100.0)
        self.video_thread.pinch_threshold = t
        self.pinch_label.setText(f"{t:.3f}")

    def on_cooldown_changed(self, value: int):
        # slider value is ms
        ms = max(50, value)
        self.video_thread.click_cooldown = ms / 1000.0
        self.cooldown_label.setText(f"{int(ms)} ms")

    def on_reset_defaults(self):
        # default values aligned with VideoThread defaults
        self.smooth_slider.setValue(int(0.2 * 100))
        self.pinch_slider.setValue(int(0.05 * 100))
        self.cooldown_slider.setValue(int(0.4 * 1000))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
