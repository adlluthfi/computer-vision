"""
Real-time Action Recognition menggunakan Webcam
Model: Hybrid LSTM (MobileNetV3Large + Pose Skeleton) with Feature Scaling
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import mediapipe as mp
from collections import deque
import time
import pickle
from pathlib import Path
import sys

class RealtimeActionRecognition:
    """Real-time action recognition system with feature scaling"""
    
    def __init__(self, model_path='model_hybrid_lstm.h5', 
                 scaler_path='scaler.pkl',
                 classes=['walking', 'jogging'],
                 sequence_length=30,
                 input_size=160,
                 grayscale_mode=True):
        
        print("="*60)
        print("🎥 REAL-TIME ACTION RECOGNITION (MobileNetV3Large)")
        print("="*60)
        
        self.input_size = input_size
        self.grayscale_mode = grayscale_mode
        
        # Load LSTM model with error handling (FIXED: Handle label smoothing)
        print("📥 Loading LSTM model...")
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load without compilation first (to avoid label smoothing issues)
            self.lstm_model = load_model(model_path, compile=False)
            
            # Recompile with label smoothing (same as training)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            
            self.lstm_model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            print("✅ LSTM model loaded and recompiled!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Trying alternative loading method...")
            try:
                # Fallback: Load with compilation
                self.lstm_model = load_model(model_path)
                print("✅ LSTM model loaded (fallback method)!")
            except Exception as e2:
                print(f"❌ Fatal error: {e2}")
                sys.exit(1)
        
        # Load scaler with validation
        print("📥 Loading feature scaler...")
        scaler_file = Path(scaler_path)
        if scaler_file.exists():
            try:
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Scaler loaded!")
                self.scaler_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading scaler: {e}")
                self.scaler_loaded = False
                self._create_dummy_scaler()
        else:
            print("⚠️ Scaler not found! Creating dummy scaler...")
            self.scaler_loaded = False
            self._create_dummy_scaler()
        
        # Load MobileNetV3Large with error handling
        print(f"📦 Loading MobileNetV3Large (Input: {input_size}x{input_size})...")
        try:
            self.mobilenet_model = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(input_size, input_size, 3)
            )
            print("✅ MobileNetV3Large loaded! Output: (1280,)")
        except Exception as e:
            print(f"❌ Error loading MobileNetV3Large: {e}")
            sys.exit(1)
        
        # Load MediaPipe Pose with error handling
        print("📦 Loading MediaPipe Pose (Lite)...")
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Pose loaded! Output: (132,)")
        except Exception as e:
            print(f"❌ Error loading MediaPipe Pose: {e}")
            sys.exit(1)
        
        self.classes = classes
        
        # DISPLAY FIX: Create display label mapping (swap jogging↔walking)
        # Model outputs: 0=jogging, 1=walking (alphabetical order from dataset)
        # But we want to display: 0=walking, 1=jogging (user expectation)
        self.display_labels = {
            0: 'jogging',  # Model's class 0 (jogging) → display as "jogging"
            1: 'walking'   # Model's class 1 (walking) → display as "walking"
        }
        
        # Reverse mapping for clarity
        self.display_to_model = {
            'walking': 1,   # Display "walking" comes from model index 1
            'jogging': 0    # Display "jogging" comes from model index 0
        }
        
        self.sequence_length = sequence_length
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # Cache for predictions
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_cache_frames = 0
        
        # Statistics
        self.total_frames_processed = 0
        self.detection_history = deque(maxlen=30)  # Last 30 predictions
        
        # Fullscreen state
        self.is_fullscreen = False
        
        print("="*60)
        print(f"Classes: {classes}")
        print(f"   ⚠️ Display mapping applied:")
        print(f"      Model index 0 → Display: {self.display_labels[0]}")
        print(f"      Model index 1 → Display: {self.display_labels[1]}")
        print(f"Sequence length: {sequence_length}")
        print(f"Input size: {input_size}x{input_size}")
        print(f"Grayscale mode: {grayscale_mode} (KTH compatibility)")  # NEW
        print(f"Hybrid features: 1280 (MobileNetV3Large) + 132 (Pose) = 1412")
        print(f"Feature scaling: {'ENABLED ✅' if self.scaler_loaded else 'DISABLED ⚠️'}")
        print(f"Label smoothing: 0.1 (training compatibility)")  # NEW
        print("="*60)
    
    def _create_dummy_scaler(self):
        """Create dummy scaler with warning"""
        print("   ⚠️ WARNING: Predictions may be inaccurate without proper scaler!")
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        dummy_data = np.random.randn(100, 1412)
        self.scaler.fit(dummy_data)
        print("   ⚠️ Using dummy scaler - please train with scaler saving!")
    
    def extract_mobilenet_features(self, frame):
        """
        Extract MobileNetV3Large features
        MODIFIED: Support grayscale mode for KTH compatibility
        """
        frame_resized = cv2.resize(frame, (self.input_size, self.input_size))
        
        if self.grayscale_mode:
            # Convert to grayscale (KTH dataset compatibility)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Histogram Equalization (enhance contrast - KTH characteristic)
            gray = cv2.equalizeHist(gray)
            
            # Convert back to RGB (3 channels required for MobileNet)
            frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            # Original color mode
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        x = image.img_to_array(frame_rgb)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = self.mobilenet_model.predict(x, verbose=0)
        return features[0]  # Shape: (1280,)
    
    def normalize_pose_landmarks(self, landmarks):
        """
        Normalize pose landmarks relative to mid-hip (NEW)
        Must match training preprocessing!
        """
        if landmarks is None or len(landmarks) != 132:
            return landmarks
        
        # MediaPipe pose indices: 23: Left Hip, 24: Right Hip
        left_hip_idx = 23 * 4  # Index 92
        right_hip_idx = 24 * 4  # Index 96
        
        # Calculate mid-hip coordinates
        mid_hip_x = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2
        mid_hip_y = (landmarks[left_hip_idx + 1] + landmarks[right_hip_idx + 1]) / 2
        
        # Normalize all x, y coordinates relative to mid-hip
        normalized = landmarks.copy()
        for i in range(0, len(normalized), 4):
            normalized[i] -= mid_hip_x      # x coordinate
            normalized[i + 1] -= mid_hip_y  # y coordinate
            # z and visibility remain unchanged
        
        return normalized
    
    def extract_pose_features(self, frame):
        """
        Extract pose keypoints
        MODIFIED: Add pose normalization (must match training!)
        """
        h, w = frame.shape[:2]
        frame_small = cv2.resize(frame, (w//2, h//2))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            
            landmarks = np.array(landmarks)  # Shape: (132,)
            
            # CRITICAL: Normalize relative to mid-hip (same as training!)
            landmarks = self.normalize_pose_landmarks(landmarks)
            
            return landmarks, results.pose_landmarks
        else:
            return np.zeros(132), None
    
    def extract_hybrid_features(self, frame):
        """Extract hybrid features with scaling"""
        mobilenet_feat = self.extract_mobilenet_features(frame)
        pose_feat, pose_landmarks = self.extract_pose_features(frame)
        
        # Concatenate
        hybrid_feat = np.concatenate([mobilenet_feat, pose_feat])
        
        # CRITICAL: Apply scaling (same as training!)
        hybrid_feat_scaled = self.scaler.transform(hybrid_feat.reshape(1, -1))[0]
        
        return hybrid_feat_scaled, pose_landmarks  # Shape: (1412,) - SCALED
    
    def predict_action(self, force=False):
        """Predict action with caching and confidence tracking"""
        if len(self.feature_buffer) < self.sequence_length:
            return None, 0.0
        
        # Use cached prediction
        if not force and self.prediction_cache_frames < 5:
            self.prediction_cache_frames += 1
            return self.last_prediction, self.last_confidence
        
        try:
            # Convert buffer to numpy array
            features = np.array(list(self.feature_buffer))
            features = np.expand_dims(features, axis=0)
            
            # Predict
            predictions = self.lstm_model.predict(features, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # DISPLAY FIX: Use display label mapping
            display_action = self.display_labels[predicted_class]
            
            # Update detection history
            self.detection_history.append({
                'action': display_action,  # CHANGED: Use mapped label
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Cache result
            self.last_prediction = display_action  # CHANGED: Use mapped label
            self.last_confidence = confidence
            self.prediction_cache_frames = 0
            
            return self.last_prediction, self.last_confidence
        except Exception as e:
            print(f"⚠️ Error during prediction: {e}")
            return None, 0.0
    
    def draw_info(self, frame, action, confidence, fps, draw_skeleton=True):
        """Draw information on frame with confidence indicator"""
        h, w = frame.shape[:2]
        
        # Adaptive UI sizing based on fullscreen mode
        if self.is_fullscreen:
            # Compact mode for fullscreen - minimal UI at top-right corner
            font_scale_title = 0.5
            font_scale_info = 0.35
            font_scale_small = 0.3
            thickness_bold = 1
            thickness_normal = 1
            padding = 8
            info_width = 280
            info_height = 110
            info_x = w - info_width - 10
            info_y = 10
        else:
            # Normal mode - larger UI at top
            font_scale_title = 0.7
            font_scale_info = 0.6
            font_scale_small = 0.5
            thickness_bold = 2
            thickness_normal = 1
            padding = 20
            info_width = w - 20
            info_height = 180
            info_x = 10
            info_y = 10
        
        # Background for text
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), (255, 255, 255), 1)
        
        # Calculate positions
        y_offset = info_y + padding
        
        if self.is_fullscreen:
            # Compact fullscreen layout
            # Title (shortened)
            cv2.putText(frame, "ACTION RECOGNITION", (info_x + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 255, 0), thickness_bold)
            y_offset += 18
            
            # Action with confidence
            if action:
                text = f"{action} {confidence*100:.0f}%"
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
                cv2.putText(frame, text, (info_x + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, color, thickness_bold)
                y_offset += 15
                
                # Compact confidence bar
                bar_width = int((info_width - 10) * confidence)
                cv2.rectangle(frame, (info_x + 5, y_offset), (info_x + 5 + bar_width, y_offset + 8), color, -1)
                cv2.rectangle(frame, (info_x + 5, y_offset), (info_x + info_width - 5, y_offset + 8), (255, 255, 255), 1)
                y_offset += 13
            else:
                cv2.putText(frame, "Collecting...", (info_x + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), thickness_normal)
                y_offset += 18
            
            # Buffer status (compact)
            buffer_pct = (len(self.feature_buffer) / self.sequence_length) * 100
            buffer_color = (0, 255, 0) if buffer_pct == 100 else (255, 255, 0)
            cv2.putText(frame, f"Buf: {len(self.feature_buffer)}/{self.sequence_length}", (info_x + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, buffer_color, thickness_normal)
            y_offset += 15
            
            # FPS
            fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps:.0f}", (info_x + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, fps_color, thickness_normal)
            
            # Compact instructions at bottom-left
            cv2.putText(frame, "q:quit | f:exit fullscreen", (10, h - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
        else:
            # Normal detailed layout
            # Title
            cv2.putText(frame, "REAL-TIME ACTION RECOGNITION", (padding, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 255, 0), thickness_bold)
            
            # Backbone info
            backbone_text = "MobileNetV3Large"
            if self.grayscale_mode:
                backbone_text += " (Grayscale)"
            cv2.putText(frame, f"Backbone: {backbone_text}", (padding, info_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 0), thickness_normal)
            
            # Action with confidence bar
            if action:
                text = f"Action: {action} ({confidence*100:.1f}%)"
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
                cv2.putText(frame, text, (padding, info_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, color, thickness_bold)
                
                # Confidence bar
                bar_width = int((w - 40) * confidence)
                cv2.rectangle(frame, (padding, info_y + 85), (padding + bar_width, info_y + 95), color, -1)
                cv2.rectangle(frame, (padding, info_y + 85), (w - padding, info_y + 95), (255, 255, 255), 1)
                
                # Low confidence warning
                if confidence < 0.5:
                    cv2.putText(frame, "⚠️ Low confidence!", (padding, info_y + 115),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 0, 255), thickness_normal)
            else:
                cv2.putText(frame, "Action: Collecting frames...", (padding, info_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), thickness_bold)
            
            # Buffer status
            buffer_pct = (len(self.feature_buffer) / self.sequence_length) * 100
            buffer_status = f"Buffer: {len(self.feature_buffer)}/{self.sequence_length} ({buffer_pct:.0f}%)"
            buffer_color = (0, 255, 0) if buffer_pct == 100 else (255, 255, 0)
            cv2.putText(frame, buffer_status, (padding, info_y + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, buffer_color, thickness_normal)
            
            # Sequence info
            cv2.putText(frame, f"Seq: {self.sequence_length} frames (~{self.sequence_length/25:.1f}s @ 25fps)", 
                       (padding, info_y + 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), thickness_normal)
            
            # FPS
            fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, fps_color, thickness_bold)
            
            # Scaler warning
            if not self.scaler_loaded:
                cv2.putText(frame, "⚠️ No scaler loaded!", (w - 200, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness_normal)
            
            # Instructions
            instructions = "q:quit | r:reset | s:toggle skeleton | f:fullscreen"
            cv2.putText(frame, instructions, (padding, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness_normal)

    def run(self, camera_id=0, feature_extract_interval=2):
        """Run real-time recognition with improved error handling"""
        
        print("\n🎬 Starting webcam...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam started!")
        print("\n⌨️  Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset buffer")
        print("   - Press 's' to toggle skeleton drawing")
        print("   - Press 'f' to toggle fullscreen")
        print(f"\n⚡ Optimization:")
        print(f"   - Feature extraction every {feature_extract_interval} frames")
        print(f"   - Sequence length: {self.sequence_length} frames (~{self.sequence_length/25:.1f}s)")
        print(f"   - Reduced input size: {self.input_size}x{self.input_size}")
        print(f"   - Grayscale mode: {self.grayscale_mode}")
        print(f"   - Lite pose model")
        print(f"   - Prediction caching")
        print("\n" + "="*60)
        
        frame_count = 0
        fps_time = time.time()
        fps = 0
        draw_skeleton = True
        last_pose_landmarks = None
        consecutive_failures = 0
        
        # Create window
        window_name = 'Real-time Action Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        print("❌ Too many consecutive frame capture failures!")
                        break
                    print(f"⚠️ Failed to grab frame (attempt {consecutive_failures}/30)")
                    continue
                
                consecutive_failures = 0  # Reset on success
                frame_count += 1
                self.total_frames_processed += 1
                
                # Extract features only every N frames
                if frame_count % feature_extract_interval == 0:
                    hybrid_features, pose_landmarks = self.extract_hybrid_features(frame)
                    self.feature_buffer.append(hybrid_features)
                    last_pose_landmarks = pose_landmarks
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - fps_time + 1e-6)  # Avoid division by zero
                fps_time = current_time
                
                # Draw pose skeleton
                if draw_skeleton and last_pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        last_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                # Predict action
                action, confidence = self.predict_action()
                
                # Draw information
                self.draw_info(frame, action, confidence, fps, draw_skeleton)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Key controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting buffer...")
                    self.feature_buffer.clear()
                    self.prediction_cache_frames = 0
                elif key == ord('s'):
                    draw_skeleton = not draw_skeleton
                    status = "ON" if draw_skeleton else "OFF"
                    print(f"\n💀 Skeleton drawing: {status}")
                elif key == ord('f'):
                    self.is_fullscreen = not self.is_fullscreen
                    if self.is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("\n🖥️  Fullscreen: ON")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("\n🖥️  Fullscreen: OFF")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            print(f"\n📊 Statistics:")
            print(f"   Total frames processed: {self.total_frames_processed}")
            print(f"   Average FPS: {fps:.1f}")
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("\n✅ Cleanup complete!")
    
    def run_video(self, video_path, feature_extract_interval=3):
        """Run on video file"""
        
        print(f"\n🎬 Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Cannot open video!")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Video loaded!")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps_video}")
        print("\n" + "="*60)
        
        frame_count = 0
        fps_time = time.time()
        fps = 0
        last_pose_landmarks = None
        
        # Create window with fullscreen support
        window_name = 'Video Action Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Extract features every N frames
                if frame_count % feature_extract_interval == 0:
                    hybrid_features, pose_landmarks = self.extract_hybrid_features(frame)
                    self.feature_buffer.append(hybrid_features)
                    last_pose_landmarks = pose_landmarks
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - fps_time)
                fps_time = current_time
                
                # Draw pose
                if last_pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        last_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                # Predict
                action, confidence = self.predict_action()
                
                # Draw info
                self.draw_info(frame, action, confidence, fps)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Key controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.is_fullscreen = not self.is_fullscreen
                    if self.is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("\n✅ Video processing complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Action Recognition (MobileNetV3Large)')
    parser.add_argument('--model', type=str, default='model_hybrid_lstm.h5',
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='scaler.pkl',
                       help='Path to feature scaler (default: scaler.pkl)')
    parser.add_argument('--classes', nargs='+', default=['walking', 'jogging'],  # CHANGED: 4 → 2 classes
                       help='List of classes (default: walking, jogging)')
    parser.add_argument('--sequence_length', type=int, default=30,  # CHANGED: 100 → 30
                       help='Sequence length (default: 30 for KTH real-time)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file')
    parser.add_argument('--input_size', type=int, default=160,
                       help='CNN input size - smaller = faster (default: 160)')
    parser.add_argument('--extract_interval', type=int, default=2,  # CHANGED: 1 → 2
                       help='Extract features every N frames (default: 2 for balance)')
    parser.add_argument('--no-grayscale', action='store_true',
                       help='Disable grayscale mode (use color)')
    
    args = parser.parse_args()
    
    # Initialize system
    recognizer = RealtimeActionRecognition(
        model_path=args.model,
        scaler_path=args.scaler,
        classes=args.classes,
        sequence_length=args.sequence_length,
        input_size=args.input_size,
        grayscale_mode=not args.no_grayscale  # NEW
    )
    
    # Run
    if args.video:
        recognizer.run_video(args.video, args.extract_interval)
    else:
        recognizer.run(args.camera, args.extract_interval)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
