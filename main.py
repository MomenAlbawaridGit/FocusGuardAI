import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time

class FocusTracker:
    def __init__(self, penalty_video_path):
        self.penalty_video_path = penalty_video_path
        
        # Load YOLOv8 Nano (Automatically uses CUDA if available)
        print("[INFO] Loading YOLO model on RTX 4060 (CUDA)...")
        self.yolo_model = YOLO('yolov8n.pt') 
        
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # State Management
        self.is_distracted = False
        self.distraction_start_time = 0
        self.GRACE_PERIOD = 3.0 # Seconds before the penalty video plays
        self.penalty_triggered = False

        # 3D Model Points for Head Pose Estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_head_pose(self, shape, frame_shape):
        image_pts = np.float32([
            shape[1], shape[152], shape[33], shape[263], shape[61], shape[291]
        ])
        
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4,1))
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Pitch (up/down), Yaw (left/right)
        pitch = angles[0] * 360
        yaw = angles[1] * 360
        return pitch, yaw

    def play_penalty_video(self):
        print("[ALERT] Focus lost. Triggering penalty video!")
        if sys.platform == "win32":
            os.startfile(self.penalty_video_path)
        elif sys.platform == "darwin": # macOS
            subprocess.call(["open", self.penalty_video_path])
        else: # Linux
            subprocess.call(["xdg-open", self.penalty_video_path])
        self.penalty_triggered = True

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Setup "Always on Top" OpenCV Window
        window_name = "FocusGuard AI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Resize and move to bottom right (assuming 1080p screen for demo)
        cv2.resizeWindow(window_name, 480, 360)
        cv2.moveWindow(window_name, 1400, 700) 

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1) # Mirror image
            h, w, c = frame.shape
            
            # 1. Phone Detection (YOLO)
            # class 67 is 'cell phone' in COCO dataset
            results = self.yolo_model.predict(frame, classes=[67], conf=0.4, verbose=False, device='cuda')
            phone_detected = len(results[0].boxes) > 0
            
            if phone_detected:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "Distraction: Phone!", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 2. Focus Tracking (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mesh = self.face_mesh.process(rgb_frame)
            
            looking_away = True
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    face_2d = []
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append((x, y))
                    
                    pitch, yaw = self.get_head_pose(face_2d, frame.shape)
                    
                    # Logic: If looking too far left/right (> 15) or down (> 10)
                    if abs(yaw) > 15 or pitch < -10:
                        looking_away = True
                        color = (0, 0, 255) # Red for looking away
                        cv2.putText(frame, "Focus Lost!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    else:
                        looking_away = False
                        color = (0, 255, 0) # Green for focused
                        cv2.putText(frame, "Focused", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                    # Draw a bounding box around face to visualize tracking
                    cv2.rectangle(frame, (face_2d[234][0], face_2d[10][1]), 
                                  (face_2d[454][0], face_2d[152][1]), color, 2)

            # 3. State Machine & Penalty Logic
            currently_distracted = phone_detected or looking_away
            
            if currently_distracted:
                if not self.is_distracted:
                    self.is_distracted = True
                    self.distraction_start_time = time.time()
                
                # Calculate how long you've been distracted
                time_distracted = time.time() - self.distraction_start_time
                cv2.putText(frame, f"Penalty in: {max(0, self.GRACE_PERIOD - time_distracted):.1f}s", 
                            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if time_distracted > self.GRACE_PERIOD and not self.penalty_triggered:
                    self.play_penalty_video()
            else:
                # Reset if you look back or put the phone down
                self.is_distracted = False
                self.penalty_triggered = False

            # Display
            cv2.imshow(window_name, frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR VIDEO
    PENALTY_VIDEO = r"C:\path\to\your\penalty_video.mp4" 
    
    app = FocusTracker(PENALTY_VIDEO)
    app.run()