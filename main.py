import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QHBoxLayout,
                             QWidget, QVBoxLayout, QPushButton, QStackedWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer

# --- Welcome Page Class ---
class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        title = QLabel("Driver Drowsiness Detection System")
        title.setFont(QFont('Arial', 28, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)

        desc = QLabel("""
        <p style='text-align: center; font-size: 16px; color: #7f8c8d;'>
        This system continuously monitors driver alertness in real-time<br>
        using advanced computer vision to detect signs of drowsiness,<br>
        including prolonged eye closure, excessive yawning, and microsleeps.
        </p>
        """)
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)

        start_btn = QPushButton("Start Detection")
        start_btn.setFont(QFont('Arial', 16))
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 25px;
                min-width: 250px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #2980b9;
                box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
        """)
        start_btn.clicked.connect(lambda: self.parent().setCurrentIndex(1))
        start_btn.setCursor(Qt.PointingHandCursor)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #ecf0f1;")


# --- DrowsinessDetector Class (Main Window, managing pages) ---
class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- IMPORTANT: Initialize Detection Variables FIRST ---
        # These need to be defined before any method potentially calls them
        self.yawn_state = ''
        self.left_eye_state =''
        self.right_eye_state= ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0
        self.eyes_closed_duration = 0
        self.blink_timestamps = []

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.alert_active = False

        # Thresholds (in seconds, can be adjusted)
        self.YAWN_THRESHOLD = 2.0
        self.MICROSLEEP_THRESHOLD = 1.2
        self.BLINK_THRESHOLD = 10
        self.BLINK_WINDOW = 30

        # --- Main Window Setup ---
        self.setWindowTitle("Driver Drowsiness Detection System")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #ecf0f1;")

        # --- Stacked Widget to manage pages ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # --- Create Pages ---
        self.welcome_page = WelcomePage()
        self.detection_page_widget = self.create_detection_page()

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.welcome_page) # Index 0
        self.stacked_widget.addWidget(self.detection_page_widget) # Index 1

        # --- Model and MediaPipe Initialization ---
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Updated points_ids to include more mouth landmarks for a better ROI.
        # These are common MediaPipe face mesh landmarks for the mouth area.
        # Ensure your model is trained on these or similar points if you adjust.
        # Original points_ids: [187, 411, 152, 68, 174, 399, 298]
        # Mouth landmarks: 187 (upper lip), 411 (right corner), 152 (lower lip) are present.
        # Adding 61 (left corner) will help define a better rectangle.
        self.points_ids = [187, 411, 152, 61, # Mouth (top, right, bottom, left)
                           68, 174, # Right Eye (inner, outer)
                           399, 298] # Left Eye (inner, outer)

        try:
            self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
            self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

        # --- Video Capture and Threading Setup ---
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Error: Could not open video device.")
            self.video_label.setText("Error: Camera not found or in use.") # Provide user feedback
            # sys.exit(1) # Don't exit immediately, let the app run without camera

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        # Threads are initialized here but started/stopped by manage_detection_threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        # --- Connect page changes to thread management ---
        self.stacked_widget.currentChanged.connect(self.manage_detection_threads)

        # Start on the welcome page
        self.stacked_widget.setCurrentIndex(0)


    def create_detection_page(self):
        """Creates and returns the widget for the detection page."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #34495e;")
        self.video_label.setFixedSize(700, 500)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera Feed Loading...")

        info_panel = QWidget()
        info_panel.setStyleSheet("background-color: white; border: 1px solid #bdc3c7; border-radius: 10px; padding: 10px;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(15, 15, 15, 15)

        self.info_label = QLabel()
        self.info_label.setFont(QFont('Arial', 12))
        self.info_label.setStyleSheet("color: #34495e;")
        self.info_label.setWordWrap(True)
        self.update_info_display() # Initial update - now safe to call

        back_btn = QPushButton("Back to Welcome")
        back_btn.setFont(QFont('Arial', 12))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        back_btn.setCursor(Qt.PointingHandCursor)

        info_layout.addWidget(self.info_label)
        info_layout.addStretch(1)
        info_layout.addWidget(back_btn)

        layout.addWidget(self.video_label, 3)
        layout.addWidget(info_panel, 1)

        return page

    def manage_detection_threads(self, index):
        """Manages starting and stopping detection threads based on the active page."""
        if index == 1: # Detection page is active
            self.start_detection()
        else: # Welcome page is active
            self.stop_detection()

    def start_detection(self):
        """Starts the video capture and processing threads."""
        print("Starting detection threads...")
        # Reset detection variables when starting
        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0
        self.eyes_closed_duration = 0
        self.blink_timestamps = []
        self.alert_active = False
        self.alert_text = ''

        self.stop_event.clear()

        # Re-initialize capture if it was released or failed to open initially
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open video device for detection.")
                self.video_label.setText("Error: Camera not found or in use.")
                return

        # Always recreate thread objects to ensure they are in a runnable state
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()
        print("Detection threads started.")


    def stop_detection(self):
        """Stops the video capture and processing threads and releases resources."""
        print("Stopping detection threads...")
        self.stop_event.set()

        # Wait for threads to finish
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1)
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1)

        if self.cap.isOpened():
            self.cap.release()
        
        self.video_label.clear()
        self.video_label.setText("Camera Feed Stopped.")
        self.update_info_display(clear=True)
        print("Detection threads stopped.")

    def closeEvent(self, event):
        """Handles the window close event to ensure threads are stopped."""
        self.stop_detection()
        super().closeEvent(event)
        event.accept()

    def update_info_display(self, clear=False):
        """Updates the info panel display with current drowsiness metrics."""
        if clear:
            info_text = (
                f"<div style='font-family: Arial, sans-serif; color: #333;'>"
                f"<h2 style='text-align: center; color: #4CAF50;'>Driver Status</h2>"
                f"<hr style='border: 1px solid #bdc3c7;'>"
                f"<p style='color: gray;'>Detection stopped.</p>"
                f"</div>"
            )
            self.info_label.setText(info_text)
            return

        current_time = time.time()

        self.blink_timestamps = [
            t for t in self.blink_timestamps
            if current_time - t <= self.BLINK_WINDOW
        ]

        recent_blinks = len(self.blink_timestamps)

        if (self.yawn_duration >= self.YAWN_THRESHOLD or
            self.eyes_closed_duration >= self.MICROSLEEP_THRESHOLD or
            recent_blinks >= self.BLINK_THRESHOLD):

            if self.yawn_duration >= self.YAWN_THRESHOLD:
                self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ ALERT: Prolonged Yawn Detected!</p>"
            elif self.eyes_closed_duration >= self.MICROSLEEP_THRESHOLD:
                self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ ALERT: Microsleep Detected!</p>"
            elif recent_blinks >= self.BLINK_THRESHOLD:
                self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ ALERT: Excessive Blinking Detected!</p>"
            self.alert_active = True
        else:
            self.alert_active = False
            self.alert_text = "<p style='color: green; font-weight: bold;'>✅ Driver Alert</p>"


        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #2c3e50;'>"
            f"<h2 style='text-align: center; color: #3498db;'>Driver Status</h2>"
            f"<hr style='border: 1px solid #bdc3c7;'>"
            f"{self.alert_text}"
            f"<p><b>👁️ Recent Blinks:</b> {recent_blinks} (last {self.BLINK_WINDOW}s)</p>"
            f"<p><b>💤 Eyes Closed:</b> {round(self.eyes_closed_duration, 2)}s</p>"
            f"<p><b>😮 Current Yawn:</b> {round(self.yawn_duration, 2)}s</p>"
            f"<p><b>🔄 Total Yawns:</b> {self.yawns}</p>"
            f"<hr style='border: 1px solid #bdc3c7;'>"
            f"<p style='font-size: 12px; color: #7f8c8d;'>"
            f"Alerts trigger after:<br>"
            f"- {self.YAWN_THRESHOLD}s yawn<br>"
            f"- {self.MICROSLEEP_THRESHOLD}s microsleep<br>"
            f"- {self.BLINK_THRESHOLD} blinks in {self.BLINK_WINDOW}s"
            f"</p>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    def predict_eye(self, eye_frame, eye_state):
        try:
            results_eye = self.detecteye.predict(eye_frame, verbose=False)
            boxes = results_eye[0].boxes
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            if class_id == 1 and confidence > 0.7:
                return "Closed"
            elif class_id == 0 and confidence > 0.7:
                return "Open"

            return eye_state
        except Exception as e:
            return eye_state


    def predict_yawn(self, yawn_frame):
        # Ensure yawn_frame is not empty or invalid
        if yawn_frame is None or yawn_frame.size == 0:
            self.yawn_state = "No Yawn"
            return

        try:
            # Suppress verbose output from YOLO
            results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
            boxes = results_yawn[0].boxes

            # If no detections, assume no yawn
            if len(boxes) == 0:
                self.yawn_state = "No Yawn"
                return

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

            # Find the detection with the highest confidence
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            # Adjust confidence thresholds if necessary after testing
            # Based on your model's training, 0 could be "Yawn" and 1 "No Yawn"
            # Verify the class mapping from your YOLO model training
            if class_id == 0 and confidence > 0.7: # Yawn class and sufficient confidence
                self.yawn_state = "Yawn"
            elif class_id == 1 and confidence > 0.6: # No Yawn class and sufficient confidence
                self.yawn_state = "No Yawn"
            else:
                # If confidence is too low for either, maintain previous state or default
                self.yawn_state = "No Yawn" # Default to no yawn if unsure

        except Exception as e:
            print(f"Yawn prediction error: {e}") # Print error for debugging
            self.yawn_state = "No Yawn" # Default to no yawn on error


    def capture_frames(self):
        """Captures frames from the camera and puts them into a queue."""
        while not self.stop_event.is_set() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < self.frame_queue.maxsize:
                    self.frame_queue.put(frame)
                else:
                    pass
            else:
                print("Failed to grab frame or camera closed.")
                break
            time.sleep(0.01)

    def process_frames(self):
        """Processes frames from the queue, performs detection, and updates UI."""
        frame_rate = 30
        frame_time = 1 / frame_rate # Use this for duration increments

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                current_processing_start_time = time.time()

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        
                        # --- Extract Landmark Coordinates ---
                        # Assuming points_ids are ordered: mouth(4), right_eye(2), left_eye(2)
                        mouth_points = []
                        for i in range(4): # First 4 points are mouth (187, 411, 152, 61)
                            lm = face_landmarks.landmark[self.points_ids[i]]
                            mouth_points.append((int(lm.x * iw), int(lm.y * ih)))

                        right_eye_points = []
                        for i in range(4, 6): # Next 2 points are right eye (68, 174)
                            lm = face_landmarks.landmark[self.points_ids[i]]
                            right_eye_points.append((int(lm.x * iw), int(lm.y * ih)))

                        left_eye_points = []
                        for i in range(6, 8): # Last 2 points are left eye (399, 298)
                            lm = face_landmarks.landmark[self.points_ids[i]]
                            left_eye_points.append((int(lm.x * iw), int(lm.y * ih)))
                        
                        # --- Calculate ROIs (more robust bounding boxes) ---
                        # Mouth ROI
                        all_mouth_x = [p[0] for p in mouth_points]
                        all_mouth_y = [p[1] for p in mouth_points]
                        x_mouth_min = max(0, min(all_mouth_x))
                        x_mouth_max = min(iw, max(all_mouth_x))
                        y_mouth_min = max(0, min(all_mouth_y))
                        y_mouth_max = min(ih, max(all_mouth_y))

                        # Right Eye ROI
                        all_reye_x = [p[0] for p in right_eye_points]
                        all_reye_y = [p[1] for p in right_eye_points]
                        x_reye_min = max(0, min(all_reye_x))
                        x_reye_max = min(iw, max(all_reye_x))
                        y_reye_min = max(0, min(all_reye_y))
                        y_reye_max = min(ih, max(all_reye_y))

                        # Left Eye ROI
                        all_leye_x = [p[0] for p in left_eye_points]
                        all_leye_y = [p[1] for p in left_eye_points]
                        x_leye_min = max(0, min(all_leye_x))
                        x_leye_max = min(iw, max(all_leye_x))
                        y_leye_min = max(0, min(all_leye_y))
                        y_leye_max = min(ih, max(all_leye_y))

                        # Check if ROIs are valid before cropping
                        if (x_mouth_max > x_mouth_min and y_mouth_max > y_mouth_min and
                            x_reye_max > x_reye_min and y_reye_max > y_reye_min and
                            x_leye_max > x_leye_min and y_leye_max > y_leye_min):

                            mouth_roi = frame[y_mouth_min:y_mouth_max, x_mouth_min:x_mouth_max]
                            right_eye_roi = frame[y_reye_min:y_reye_max, x_reye_min:x_reye_max]
                            left_eye_roi = frame[y_leye_min:y_leye_max, x_leye_min:x_leye_max]

                            self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                            self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                            self.predict_yawn(mouth_roi) # This is where yawn prediction happens

                            # --- Drowsiness Logic Update ---
                            both_eyes_closed = (self.left_eye_state == "Closed" and
                                                self.right_eye_state == "Closed")

                            if both_eyes_closed:
                                if not (self.left_eye_still_closed and self.right_eye_still_closed):
                                    self.left_eye_still_closed = True
                                    self.right_eye_still_closed = True
                                    self.blinks += 1
                                    self.blink_timestamps.append(time.time())
                                self.eyes_closed_duration += frame_time # Use frame_time
                            else:
                                if self.left_eye_still_closed and self.right_eye_still_closed:
                                    self.left_eye_still_closed = False
                                    self.right_eye_still_closed = False
                                self.eyes_closed_duration = 0

                            if self.yawn_state == "Yawn":
                                if not self.yawn_in_progress:
                                    self.yawn_in_progress = True
                                    self.yawns += 1
                                self.yawn_duration += frame_time # Use frame_time
                            else:
                                if self.yawn_in_progress:
                                    self.yawn_in_progress = False
                                self.yawn_duration = 0

                            # Play alert sound if active and not already playing
                            if self.alert_active and (not hasattr(self, '_sound_thread') or not self._sound_thread.is_alive()):
                                self.play_sound_in_thread()

                            # Draw annotations on the frame
                            all_points = mouth_points + right_eye_points + left_eye_points
                            self.draw_annotations(frame, all_points)

                self.update_info_display()
                self.display_frame(frame)

                # Control frame processing speed
                processing_time = time.time() - current_processing_start_time
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)

            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)

    def draw_annotations(self, frame, points):
        """Draws landmarks, eye state, and yawn state on the frame."""
        for (x, y) in points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        eye_state_text = f"L Eye: {self.left_eye_state}, R Eye: {self.right_eye_state}"
        cv2.putText(frame, eye_state_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        yawn_text = f"Yawn: {self.yawn_state}"
        cv2.putText(frame, yawn_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.alert_active:
            cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    def display_frame(self, frame):
        """Converts an OpenCV frame to QPixmap and displays it on video_label."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
        """Plays an alert sound."""
        try:
            for _ in range(3):
                winsound.Beep(1000, 200)
                time.sleep(0.1)
        except Exception as e:
            print(f"Sound error: {e}")

    def play_sound_in_thread(self):
        """Starts a new thread to play the alert sound."""
        self._sound_thread = threading.Thread(target=self.play_alert_sound)
        self._sound_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DrowsinessDetector()
    window.show()

    sys.exit(app.exec_())
