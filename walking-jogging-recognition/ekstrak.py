import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import mediapipe as mp
from sklearn.model_selection import train_test_split

class HybridFeatureExtractor:
    """Extract hybrid features: MobileNetV3Large (spatial) + Pose Skeleton (structural)"""
    
    def __init__(self, enable_augmentation=True, augmentation_factor=2, grayscale_mode=True):
        # Load MobileNetV3Large untuk spatial features
        print("📦 Loading MobileNetV3Large (pre-trained ImageNet)...")
        self.mobilenet_model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        print("✅ MobileNetV3Large loaded! Output: (1280,)")
        
        # Load MediaPipe Pose untuk skeleton
        print("📦 Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Pose loaded! Output: 33 keypoints x 4 = (132,)")
        
        # Augmentation settings
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        self.grayscale_mode = grayscale_mode  # KTH dataset is grayscale
        
        print("="*60)
        print("🔗 Hybrid Features: 1280 (MobileNetV3Large) + 132 (Pose) = 1412 dimensions")
        
        if grayscale_mode:
            print("="*60)
            print("⚙️ GRAYSCALE MODE: ENABLED (KTH Dataset Compatibility)")
            print("   - Convert to grayscale + histogram equalization")
            print("   - RGB conversion for MobileNet compatibility")
            print("   - Focus on structural features, not color")
        
        if enable_augmentation:
            print("="*60)
            print("✅ DATA AUGMENTATION: ENABLED")
            print(f"   Augmentation factor: {augmentation_factor}x (each video → {augmentation_factor} variations)")
            print("   Temporal augmentations:")
            print("   1. Speed variation: 0.8x (slower), 1.0x (normal), 1.2x (faster)")
            print("   2. Random temporal crop: 80-100% of frames")
            print("   3. Frame sampling: every 1-2 frames")
            print("   4. Time shift: ±3–5 frames")
            print("   5. Gaussian noise: σ 0.005–0.01")
            print("   6. Feature dropout: 5%")
            print("   7. Temporal masking: ≤5%")
            print("   8. Horizontal flip (pose landmarks)")  # NEW
            print("   Benefits:")
            print("   - Increased dataset size by {}x".format(augmentation_factor))
            print("   - Better generalization")
            print("   - Robust to speed variations")
            print("   - Direction-invariant pose features")  # NEW
        else:
            print("⚠️  DATA AUGMENTATION: DISABLED")
        
        print("="*60)
    
    def augment_temporal_speed(self, features, speed_factor):
        """Augment by changing playback speed (temporal resampling)"""
        if abs(speed_factor - 1.0) < 0.01:  # Normal speed
            return features
        
        num_frames = len(features)
        new_num_frames = int(num_frames / speed_factor)
        
        if new_num_frames < 5:
            return features
        
        # Linear interpolation for resampling
        indices = np.linspace(0, num_frames - 1, new_num_frames)
        
        # Interpolate each feature dimension
        augmented = np.zeros((new_num_frames, features.shape[1]))
        for i in range(features.shape[1]):
            augmented[:, i] = np.interp(indices, np.arange(num_frames), features[:, i])
        
        return augmented
    
    def augment_random_crop(self, features, min_ratio=0.8):
        """Random temporal crop (simulate different video lengths)"""
        num_frames = len(features)
        crop_ratio = np.random.uniform(min_ratio, 1.0)
        crop_length = max(5, int(num_frames * crop_ratio))
        
        if crop_length >= num_frames:
            return features
        
        start_idx = np.random.randint(0, num_frames - crop_length + 1)
        return features[start_idx:start_idx + crop_length]
    
    def augment_frame_sampling(self, features, sampling_rate):
        """Sample every N frames (simulate different frame rates)"""
        if sampling_rate <= 1:
            return features
        
        sampled = features[::sampling_rate]
        
        if len(sampled) < 5:
            return features
        
        return sampled
    
    def augment_time_shift(self, features, shift_range=(-5, 5)):
        """Time shift: shift sequence by ±3–5 frames (circular shift)"""
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        if shift == 0:
            return features
        return np.roll(features, shift, axis=0)
    
    def augment_gaussian_noise(self, features, sigma_range=(0.005, 0.01)):
        """Add small Gaussian noise: σ 0.005–0.01"""
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, features.shape)
        return features + noise
    
    def augment_feature_dropout(self, features, dropout_rate=0.05):
        """Feature dropout: randomly zero out 5% of features"""
        mask = np.random.random(features.shape) > dropout_rate
        return features * mask
    
    def augment_temporal_masking(self, features, mask_ratio=0.05):
        """Temporal masking: mask up to 5% of frames"""
        num_frames = len(features)
        mask_length = max(1, int(num_frames * mask_ratio))
        
        if mask_length >= num_frames:
            return features
        
        # Random start position for masking
        start_idx = np.random.randint(0, num_frames - mask_length + 1)
        
        # Create a copy and mask the selected frames (set to zero)
        masked_features = features.copy()
        masked_features[start_idx:start_idx + mask_length] = 0
        
        return masked_features
    
    def augment_horizontal_flip(self, features):
        """
        Horizontal flip for pose features (NEW)
        Flips x-coordinates of pose landmarks (features[1280:])
        MobileNet features (features[:1280]) remain unchanged
        """
        augmented = features.copy()
        
        # Pose features start at index 1280
        pose_features = augmented[1280:]
        
        # Each landmark has 4 values: x, y, z, visibility
        # Flip only x-coordinates (every 4th value starting from 0)
        for i in range(0, len(pose_features), 4):
            pose_features[i] = 1.0 - pose_features[i]  # Flip x: 0.3 → 0.7
        
        augmented[1280:] = pose_features
        return augmented
    
    def apply_augmentation(self, features, aug_type):
        """Apply specific augmentation type"""
        if aug_type == 'original':
            return features
        elif aug_type == 'speed_slow':
            return self.augment_temporal_speed(features, 0.8)
        elif aug_type == 'speed_fast':
            return self.augment_temporal_speed(features, 1.2)
        elif aug_type == 'crop':
            return self.augment_random_crop(features)
        elif aug_type == 'sample':
            return self.augment_frame_sampling(features, 2)
        elif aug_type == 'time_shift':
            return self.augment_time_shift(features, shift_range=(-5, 5))
        elif aug_type == 'gaussian_noise':
            return self.augment_gaussian_noise(features, sigma_range=(0.005, 0.01))
        elif aug_type == 'feature_dropout':
            return self.augment_feature_dropout(features, dropout_rate=0.05)
        elif aug_type == 'temporal_masking':
            return self.augment_temporal_masking(features, mask_ratio=0.05)
        elif aug_type == 'horizontal_flip':  # NEW
            return np.array([self.augment_horizontal_flip(f) for f in features])
        else:
            return features
    
    def extract_mobilenet_features(self, frame):
        """
        Extract MobileNetV3Large features
        MODIFIED: KTH dataset is grayscale - apply histogram equalization
        """
        frame_resized = cv2.resize(frame, (224, 224))
        
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
        Makes model focus on movement patterns, not absolute position
        
        Args:
            landmarks: Raw landmarks array (132,) - 33 keypoints x 4 values
        
        Returns:
            Normalized landmarks array (132,)
        """
        if landmarks is None or len(landmarks) != 132:
            return landmarks
        
        # MediaPipe pose indices:
        # 23: Left Hip, 24: Right Hip
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
        MODIFIED: Add pose normalization
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            # Normalize relative to mid-hip
            landmarks = self.normalize_pose_landmarks(landmarks)
            
            return landmarks
        else:
            return np.zeros(132)
    
    def validate_pose_sequence(self, features, zero_threshold=0.7):  # CHANGED: 0.5 → 0.7 (more lenient)
        """
        Validate pose sequence quality (IMPROVED)
        Reject sequences with too many zero frames (pose not detected)
        
        Args:
            features: Feature array (num_frames, 1412)
            zero_threshold: Maximum ratio of zero frames allowed (default: 70%)
        
        Returns:
            bool: True if valid, False if too many zero frames
        """
        if features is None or len(features) == 0:
            return False
        
        # Count frames where pose is all zeros (1280:1412)
        pose_features = features[:, 1280:]  # Extract pose part
        zero_frames = np.all(pose_features == 0, axis=1).sum()
        zero_ratio = zero_frames / len(features)
        
        # DIAGNOSTIC: Print validation info
        # print(f"      Pose validation: {zero_frames}/{len(features)} zero frames ({zero_ratio*100:.1f}%)")
        
        if zero_ratio > zero_threshold:
            return False
        
        return True
    
    def extract_hybrid_features(self, frame):
        """Extract hybrid features"""
        mobilenet_feat = self.extract_mobilenet_features(frame)
        pose_feat = self.extract_pose_features(frame)
        
        # Concatenate
        hybrid_feat = np.concatenate([mobilenet_feat, pose_feat])
        return hybrid_feat  # Shape: (1412,)
    
    def extract_video_features_batch(self, video_path, max_frames=None, batch_size=32):
        """
        Extract hybrid features from video with batch processing (OPTIMIZED)
        Process MobileNet predictions in batches for better performance
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
            batch_size: Batch size for MobileNet inference
        
        Returns:
            Feature array (num_frames, 1412)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"   ❌ Cannot open: {video_path.name}")
            return None
        
        # Collect frames first
        frames_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_list.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        if len(frames_list) == 0:
            return None
        
        # Process MobileNet features in batches
        mobilenet_features_list = []
        
        for i in range(0, len(frames_list), batch_size):
            batch_frames = frames_list[i:i + batch_size]
            
            # Prepare batch
            batch_input = []
            for frame in batch_frames:
                frame_resized = cv2.resize(frame, (224, 224))
                
                if self.grayscale_mode:
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                x = image.img_to_array(frame_rgb)
                batch_input.append(x)
            
            # Batch prediction
            batch_input = np.array(batch_input)
            batch_input = preprocess_input(batch_input)
            batch_features = self.mobilenet_model.predict(batch_input, verbose=0)
            
            mobilenet_features_list.extend(batch_features)
        
        # Extract pose features (sequential - cannot batch easily)
        pose_features_list = []
        for frame in frames_list:
            pose_feat = self.extract_pose_features(frame)
            pose_features_list.append(pose_feat)
        
        # Combine features
        features_list = []
        for mob_feat, pose_feat in zip(mobilenet_features_list, pose_features_list):
            hybrid_feat = np.concatenate([mob_feat, pose_feat])
            features_list.append(hybrid_feat)
        
        return np.array(features_list)  # Shape: (num_frames, 1412)
    
    def extract_video_features(self, video_path, max_frames=None):
        """Extract hybrid features dari video (use batch version for efficiency)"""
        return self.extract_video_features_batch(video_path, max_frames, batch_size=32)
    
    def split_dataset(self, video_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """Split dataset into train, validation, and test sets"""
        # Pastikan ratio = 100%
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle dengan random seed yang sama untuk reproducibility
        np.random.seed(random_seed)
        video_files = list(video_files)
        np.random.shuffle(video_files)
        
        total = len(video_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_files = video_files[:train_size]
        val_files = video_files[train_size:train_size + val_size]
        test_files = video_files[train_size + val_size:]
        
        return train_files, val_files, test_files
    
    def merge_dataset(self, output_dir='dataset', output_file='merged_dataset.npz', 
                      sequence_length=100, padding='post'):
        """
        Merge all extracted features into single dataset files
        
        Args:
            output_dir: Directory containing train/val/test splits
            output_file: Output filename pattern (will create train_, val_, test_ prefixes)
            sequence_length: Fixed sequence length (pad/truncate)
            padding: 'pre' or 'post' padding
        
        Returns:
            Dictionary with statistics
        """
        print("="*60)
        print("📦 MERGING DATASET INTO SINGLE FILES")
        print("="*60)
        print(f"Source directory: {output_dir}")
        print(f"Sequence length: {sequence_length}")
        print(f"Padding mode: {padding}")
        print("="*60)
        
        output_path = Path(output_dir)
        stats = {}
        
        for split in ['train', 'val', 'test']:
            split_path = output_path / split
            
            if not split_path.exists():
                print(f"\n⚠️ {split} directory not found, skipping...")
                continue
            
            print(f"\n📂 Processing {split} split...")
            
            X_list = []
            y_list = []
            file_list = []
            
            # Get all .npy files
            npy_files = list(split_path.rglob('*_hybrid*.npy'))
            
            if len(npy_files) == 0:
                print(f"   ⚠️ No .npy files found in {split}")
                continue
            
            # Determine class labels
            classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
            class_to_idx = {c: i for i, c in enumerate(classes)}
            
            print(f"   Classes: {classes}")
            print(f"   Files to process: {len(npy_files)}")
            
            for npy_file in tqdm(npy_files, desc=f"  Loading {split}"):
                # Get class label from parent directory
                class_name = npy_file.parent.name
                
                if class_name not in class_to_idx:
                    continue
                
                # Load features
                try:
                    features = np.load(npy_file)
                    
                    # Pad or truncate to sequence_length
                    if len(features) < sequence_length:
                        # Padding
                        pad_width = sequence_length - len(features)
                        if padding == 'post':
                            features = np.vstack([features, np.zeros((pad_width, features.shape[1]))])
                        else:  # pre
                            features = np.vstack([np.zeros((pad_width, features.shape[1])), features])
                    elif len(features) > sequence_length:
                        # Truncate (center crop)
                        start = (len(features) - sequence_length) // 2
                        features = features[start:start + sequence_length]
                    
                    X_list.append(features)
                    y_list.append(class_to_idx[class_name])
                    file_list.append(npy_file.name)
                    
                except Exception as e:
                    print(f"   ⚠️ Error loading {npy_file.name}: {e}")
                    continue
            
            if len(X_list) == 0:
                print(f"   ⚠️ No valid data in {split}")
                continue
            
            # Convert to numpy arrays
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Save merged dataset
            output_filename = output_path / f"{split}_{output_file}"
            np.savez_compressed(
                output_filename,
                X=X,
                y=y,
                classes=classes,
                files=file_list
            )
            
            stats[split] = {
                'samples': len(X),
                'shape': X.shape,
                'classes': len(classes),
                'file': str(output_filename)
            }
            
            print(f"   ✅ Saved: {output_filename}")
            print(f"      Shape: {X.shape}")
            print(f"      Labels: {len(y)} ({len(classes)} classes)")
            print(f"      File size: {output_filename.stat().st_size / 1024 / 1024:.2f} MB")
        
        print("\n" + "="*60)
        print("✅ DATASET MERGE COMPLETE")
        print("="*60)
        
        for split, info in stats.items():
            print(f"{split.upper()}:")
            print(f"  Samples: {info['samples']}")
            print(f"  Shape: {info['shape']}")
            print(f"  Classes: {info['classes']}")
            print(f"  File: {info['file']}")
        
        print("\n💡 Usage in training:")
        print("   data = np.load('dataset/train_merged_dataset.npz')")
        print("   X_train = data['X']")
        print("   y_train = data['y']")
        print("   classes = data['classes']")
        
        return stats
    
    def process_dataset(self, classes=['walking', 'walking1', 'jogging', 'jogging1'], 
                       max_frames=100, split_data=True, output_dir='dataset',
                       merge_files=False, sequence_length=100, skip_pose_validation=False):
        """Process semua video dengan split train/val/test dan augmentasi"""
        
        print("="*60)
        print("🎬 EXTRACTING HYBRID FEATURES (MobileNetV3Large + Pose)")
        print("="*60)
        print(f"Classes: {classes}")
        print(f"Max frames per video: {max_frames}")
        print(f"Split data: {split_data}")
        print(f"Grayscale mode: {self.grayscale_mode} (KTH compatibility)")
        print(f"Pose validation: {'DISABLED' if skip_pose_validation else 'ENABLED (70% threshold)'}")  # NEW
        if split_data:
            print(f"Output structure: {output_dir}/train, {output_dir}/val, {output_dir}/test")
            print(f"Split ratio: 80% train, 10% val, 10% test")
        if self.enable_augmentation:
            print(f"Augmentation: ENABLED ({self.augmentation_factor}x)")
        print("="*60)
        
        # Create output directories
        if split_data:
            output_path = Path(output_dir)
            for split in ['train', 'val', 'test']:
                for class_name in classes:
                    (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        total_stats = {'train': 0, 'val': 0, 'test': 0}
        skipped_stats = {'insufficient_frames': 0, 'pose_validation': 0, 'total_processed': 0}  # NEW
        
        for class_name in classes:
            class_dir = Path(class_name)
            
            if not class_dir.exists():
                print(f"\n⚠️ Folder {class_name}/ not found, skip")
                continue
            
            # IMPROVED: Case-insensitive search with deduplication
            video_files = set()  # Use set to avoid duplicates
            
            # Search for video files (case-insensitive)
            for pattern in ['*.mp4', '*.avi', '*.mov']:
                # Lowercase pattern
                video_files.update(class_dir.glob(pattern))
                # Uppercase pattern
                video_files.update(class_dir.glob(pattern.upper()))
            
            # Convert set to list and filter non-hybrid files
            video_files = [
                v for v in video_files 
                if not v.stem.endswith('_hybrid') and not v.stem.endswith('_HYBRID')
            ]
            
            # Sort for consistent ordering
            video_files = sorted(video_files, key=lambda x: x.name.lower())
            
            if len(video_files) == 0:
                print(f"\n⚠️ No videos in {class_name}/, skip")
                continue
            
            print(f"\n📂 {class_name}: {len(video_files)} videos")
            
            # Split dataset
            if split_data:
                train_files, val_files, test_files = self.split_dataset(video_files)
                splits = {
                    'train': train_files,
                    'val': val_files,
                    'test': test_files
                }
                print(f"   Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
            else:
                splits = {'all': video_files}
            
            # Process each split
            for split_name, files in splits.items():
                # Define augmentation types based on split
                if self.enable_augmentation and split_name == 'train':
                    # Apply augmentation only to training data
                    aug_types = ['original', 'speed_slow', 'speed_fast', 'crop', 'sample', 
                                'time_shift', 'gaussian_noise', 'feature_dropout', 
                                'temporal_masking', 'horizontal_flip'][:self.augmentation_factor + 1]
                else:
                    # No augmentation for val/test
                    aug_types = ['original']
                
                for video_file in tqdm(files, desc=f"  {class_name}/{split_name}"):
                    skipped_stats['total_processed'] += 1
                    
                    # Extract original features (with batch processing)
                    features = self.extract_video_features(video_file, max_frames=max_frames)
                    
                    if features is None or len(features) < 5:
                        # print(f"   ⚠️ Skipping {video_file.name}: insufficient frames")
                        skipped_stats['insufficient_frames'] += 1
                        continue
                    
                    # Validate pose quality (OPTIONAL)
                    if not skip_pose_validation:
                        if not self.validate_pose_sequence(features, zero_threshold=0.7):  # More lenient
                            # print(f"   ⚠️ Skipping {video_file.name}: too many frames without pose detection (>70%)")
                            skipped_stats['pose_validation'] += 1
                            continue
                    
                    # Save original + augmented versions
                    for aug_idx, aug_type in enumerate(aug_types):
                        # Apply augmentation
                        augmented_features = self.apply_augmentation(features, aug_type)
                        
                        if len(augmented_features) < 5:
                            continue
                        
                        # Determine output path
                        if split_data:
                            if aug_type == 'original':
                                output_file = Path(output_dir) / split_name / class_name / f"{video_file.stem}_hybrid.npy"
                            else:
                                output_file = Path(output_dir) / split_name / class_name / f"{video_file.stem}_hybrid_{aug_type}.npy"
                        else:
                            if aug_type == 'original':
                                output_file = video_file.parent / f"{video_file.stem}_hybrid.npy"
                            else:
                                output_file = video_file.parent / f"{video_file.stem}_hybrid_{aug_type}.npy"
                        
                        # Skip if already processed
                        if output_file.exists():
                            # print(f"   ℹ️ Skipping {output_file.name}: already exists")
                            continue
                        
                        # Save features
                        np.save(output_file, augmented_features)
                        total_stats[split_name] += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Hybrid feature extraction complete!")
        if split_data:
            print(f"   Train: {total_stats['train']} files (with augmentation)")
            print(f"   Val: {total_stats['val']} files (original only)")
            print(f"   Test: {total_stats['test']} files (original only)")
            print(f"   Total: {sum(total_stats.values())} files")
            if self.enable_augmentation and total_stats['train'] > 0:
                original_train = total_stats['train'] // (self.augmentation_factor + 1)
                print(f"   Train augmentation: ~{original_train} original → {total_stats['train']} total")
        else:
            print(f"   Total: {total_stats.get('all', 0)} files processed")
        
        # Show skip statistics
        print(f"\n📊 Processing Statistics:")
        print(f"   Videos processed: {skipped_stats['total_processed']}")
        print(f"   Skipped (insufficient frames): {skipped_stats['insufficient_frames']}")
        print(f"   Skipped (pose validation): {skipped_stats['pose_validation']}")
        success_count = skipped_stats['total_processed'] - skipped_stats['insufficient_frames'] - skipped_stats['pose_validation']
        print(f"   Successfully extracted: {success_count}")
        
        if success_count == 0:
            print(f"\n⚠️ WARNING: No videos were successfully extracted!")
            print(f"   Try running with: --skip-pose-validation")
        
        print(f"   Features: 1280 (MobileNetV3Large) + 132 (Pose) = 1412")
        print(f"{'='*60}")
        
        # Merge dataset if requested
        if merge_files:
            print("\n")
            self.merge_dataset(
                output_dir=output_dir,
                output_file='merged_dataset.npz',
                sequence_length=sequence_length
            )
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


def process_merged_classes(extractor, output_dir='dataset', max_frames=40, 
                           merge_files=True, sequence_length=30, skip_pose_validation=True):
    """
    Process dataset with merged classes (walking1→walking, jogging1→jogging)
    FIXED: Split BEFORE augmentation to maintain proper 80/10/10 ratio
    """
    
    # Define class mapping
    class_mapping = {
        'walking': ['walking', 'walking1'],
        'jogging': ['jogging', 'jogging1']
    }
    
    final_classes = ['walking', 'jogging']
    
    print("="*60)
    print("🎬 EXTRACTING HYBRID FEATURES (2-Class Mode)")
    print("="*60)
    print(f"Target classes: {final_classes}")
    print(f"Max frames per video: {max_frames}")
    print(f"Grayscale mode: {extractor.grayscale_mode} (KTH compatibility)")
    print(f"Pose validation: {'DISABLED' if skip_pose_validation else 'ENABLED (70% threshold)'}")
    print(f"Output structure: {output_dir}/train, {output_dir}/val, {output_dir}/test")
    print(f"Split ratio: 80% train, 10% val, 10% test")
    if extractor.enable_augmentation:
        print(f"Augmentation: ENABLED ({extractor.augmentation_factor}x)")
    print("="*60)
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for class_name in final_classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    skipped_stats = {'insufficient_frames': 0, 'pose_validation': 0, 'total_processed': 0}
    
    # Process each merged class
    for target_class, source_classes in class_mapping.items():
        print(f"\n📂 Processing {target_class} (from {', '.join(source_classes)})")
        
        # Collect all videos from source folders
        all_videos = []
        for source_class in source_classes:
            class_dir = Path(source_class)
            
            if not class_dir.exists():
                print(f"   ⚠️ Folder {source_class}/ not found, skip")
                continue
            
            # Case-insensitive search with deduplication
            video_files = set()
            for pattern in ['*.mp4', '*.avi', '*.mov']:
                video_files.update(class_dir.glob(pattern))
                video_files.update(class_dir.glob(pattern.upper()))
            
            video_files = [
                v for v in video_files 
                if not v.stem.endswith('_hybrid') and not v.stem.endswith('_HYBRID')
            ]
            
            all_videos.extend(video_files)
            print(f"   Found {len(video_files)} videos in {source_class}/")
        
        if len(all_videos) == 0:
            print(f"   ⚠️ No videos found for {target_class}, skip")
            continue
        
        # Sort for consistent ordering
        all_videos = sorted(all_videos, key=lambda x: x.name.lower())
        
        print(f"   Total: {len(all_videos)} videos for class '{target_class}'")
        
        # CRITICAL FIX: Split dataset FIRST (before extraction)
        train_files, val_files, test_files = extractor.split_dataset(all_videos)
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # Show split info
        print(f"   Split (before augmentation):")
        print(f"   - Train: {len(train_files)} videos ({len(train_files)/len(all_videos)*100:.1f}%)")
        print(f"   - Val: {len(val_files)} videos ({len(val_files)/len(all_videos)*100:.1f}%)")
        print(f"   - Test: {len(test_files)} videos ({len(test_files)/len(all_videos)*100:.1f}%)")
        
        # Process each split
        for split_name, files in splits.items():
            # CRITICAL: Augmentation only for TRAIN split
            if extractor.enable_augmentation and split_name == 'train':
                aug_types = ['original', 'speed_slow', 'speed_fast', 'crop', 'sample', 
                            'time_shift', 'gaussian_noise', 'feature_dropout', 
                            'temporal_masking', 'horizontal_flip'][:extractor.augmentation_factor + 1]
                print(f"   Train augmentation: {len(files)} → {len(files) * len(aug_types)} files ({len(aug_types)}x)")
            else:
                # NO augmentation for val/test
                aug_types = ['original']
            
            for video_file in tqdm(files, desc=f"  {target_class}/{split_name}"):
                skipped_stats['total_processed'] += 1
                
                # Extract features
                features = extractor.extract_video_features(video_file, max_frames=max_frames)
                
                if features is None or len(features) < 5:
                    skipped_stats['insufficient_frames'] += 1
                    continue
                
                # Validate pose quality
                if not skip_pose_validation:
                    if not extractor.validate_pose_sequence(features, zero_threshold=0.7):
                        skipped_stats['pose_validation'] += 1
                        continue
                
                # Save original + augmented versions
                for aug_type in aug_types:
                    augmented_features = extractor.apply_augmentation(features, aug_type)
                    
                    if len(augmented_features) < 5:
                        continue
                    
                    # Output path uses target_class (not source folder)
                    if aug_type == 'original':
                        output_file = output_path / split_name / target_class / f"{video_file.stem}_hybrid.npy"
                    else:
                        output_file = output_path / split_name / target_class / f"{video_file.stem}_hybrid_{aug_type}.npy"
                    
                    if output_file.exists():
                        continue
                    
                    np.save(output_file, augmented_features)
                    total_stats[split_name] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"✅ Hybrid feature extraction complete!")
    
    # Calculate actual vs expected split ratios
    total_files = sum(total_stats.values())
    if total_files > 0:
        train_pct = total_stats['train'] / total_files * 100
        val_pct = total_stats['val'] / total_files * 100
        test_pct = total_stats['test'] / total_files * 100
        
        print(f"   Train: {total_stats['train']} files ({train_pct:.1f}%)")
        print(f"   Val: {total_stats['val']} files ({val_pct:.1f}%)")
        print(f"   Test: {total_stats['test']} files ({test_pct:.1f}%)")
        print(f"   Total: {total_files} files")
        
        # Show original video count vs augmented file count
        if extractor.enable_augmentation and total_stats['train'] > 0:
            aug_factor = extractor.augmentation_factor + 1
            original_train = total_stats['train'] // aug_factor
            print(f"\n   📊 Augmentation Impact:")
            print(f"   - Original train videos: {original_train}")
            print(f"   - Augmented train files: {total_stats['train']} ({aug_factor}x)")
            print(f"   - Val/Test: NO augmentation (original only)")
            
            # Verify split ratio based on ORIGINAL videos
            original_total = original_train + total_stats['val'] + total_stats['test']
            print(f"\n   ✅ Original Split Ratio (before augmentation):")
            print(f"   - Train: {original_train}/{original_total} ({original_train/original_total*100:.1f}%)")
            print(f"   - Val: {total_stats['val']}/{original_total} ({total_stats['val']/original_total*100:.1f}%)")
            print(f"   - Test: {total_stats['test']}/{original_total} ({total_stats['test']/original_total*100:.1f}%)")
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Videos processed: {skipped_stats['total_processed']}")
    print(f"   Skipped (insufficient frames): {skipped_stats['insufficient_frames']}")
    print(f"   Skipped (pose validation): {skipped_stats['pose_validation']}")
    success_count = skipped_stats['total_processed'] - skipped_stats['insufficient_frames'] - skipped_stats['pose_validation']
    print(f"   Successfully extracted: {success_count}")
    
    if success_count == 0:
        print(f"\n⚠️ WARNING: No videos were successfully extracted!")
        print(f"   Try running with: --skip-pose-validation")
    
    print(f"   Features: 1280 (MobileNetV3Large) + 132 (Pose) = 1412")
    print(f"{'='*60}")
    
    # Merge dataset
    if merge_files:
        print("\n")
        extractor.merge_dataset(
            output_dir=output_dir,
            output_file='merged_dataset.npz',
            sequence_length=sequence_length
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract hybrid features (MobileNetV3Large + Pose) with Augmentation')
    parser.add_argument('--classes', nargs='+', default=['walking', 'jogging'],
                       help='List of class folders (default: walking, jogging)')
    parser.add_argument('--merge_classes', action='store_true',
                       help='Merge walking1→walking, jogging1→jogging into 2 classes')
    parser.add_argument('--max_frames', type=int, default=40,
                       help='Max frames per video (default: 40 for KTH real-time)')
    parser.add_argument('--no-split', action='store_true',
                       help='Do not split into train/val/test')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for split data (default: dataset)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=2,
                       help='Augmentation factor: 2=original+1aug, 3=original+2aug, etc (default: 2)')
    parser.add_argument('--list-files', action='store_true',
                       help='List all video files without processing')
    parser.add_argument('--merge', action='store_true',
                       help='Merge all features into single dataset files (train/val/test)')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Fixed sequence length for merged dataset (default: 30 for KTH)')
    parser.add_argument('--no-grayscale', action='store_true',
                       help='Disable grayscale mode (use color)')
    parser.add_argument('--merge-only', action='store_true',
                       help='Only merge existing .npy files (skip extraction)')
    parser.add_argument('--skip-pose-validation', action='store_true',
                       help='Skip pose quality validation (extract all videos)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed extraction info for each video')
    
    args = parser.parse_args()
    
    # Auto-enable merge_classes if using default 2-class setup
    if args.classes == ['walking', 'jogging']:
        args.merge_classes = True
        print("ℹ️ Using 2-class mode: walking + walking1 → walking, jogging + jogging1 → jogging")
    
    # List files mode
    if args.list_files:
        print("📋 Listing video files...")
        
        if args.merge_classes:
            # Map for merged classes
            class_mapping = {
                'walking': ['walking', 'walking1'],
                'jogging': ['jogging', 'jogging1']
            }
            
            for target_class, source_classes in class_mapping.items():
                total_files = []
                for source_class in source_classes:
                    class_dir = Path(source_class)
                    if not class_dir.exists():
                        continue
                    
                    video_files = set()
                    for pattern in ['*.mp4', '*.avi', '*.mov']:
                        video_files.update(class_dir.glob(pattern))
                        video_files.update(class_dir.glob(pattern.upper()))
                    
                    video_files = [
                        v for v in video_files 
                        if not v.stem.endswith('_hybrid') and not v.stem.endswith('_HYBRID')
                    ]
                    total_files.extend(video_files)
                
                print(f"\n{target_class}: {len(total_files)} files (from {', '.join(source_classes)})")
                for i, vf in enumerate(sorted(total_files, key=lambda x: x.name.lower()), 1):
                    print(f"  {i}. {vf.name} [{vf.parent.name}]")
        else:
            for class_name in args.classes:
                class_dir = Path(class_name)
                if not class_dir.exists():
                    continue
                
                video_files = set()
                for pattern in ['*.mp4', '*.avi', '*.mov']:
                    video_files.update(class_dir.glob(pattern))
                    video_files.update(class_dir.glob(pattern.upper()))
                
                video_files = [v for v in video_files if not v.stem.endswith('_hybrid')]
                
                print(f"\n{class_name}: {len(video_files)} files")
                for i, vf in enumerate(sorted(video_files), 1):
                    print(f"  {i}. {vf.name}")
        return
    
    extractor = HybridFeatureExtractor(
        enable_augmentation=not args.no_augmentation,
        augmentation_factor=args.augmentation_factor,
        grayscale_mode=not args.no_grayscale
    )
    
    # Merge-only mode
    if args.merge_only:
        print("📦 MERGE-ONLY MODE")
        extractor.merge_dataset(
            output_dir=args.output_dir,
            output_file='merged_dataset.npz',
            sequence_length=args.sequence_length
        )
        return
    
    # Process dataset with class merging
    if args.merge_classes:
        print("\n" + "="*60)
        print("📁 CLASS MERGING MODE")
        print("="*60)
        print("   walking + walking1 → walking (2 folders → 1 class)")
        print("   jogging + jogging1 → jogging (2 folders → 1 class)")
        print("   Final classes: ['walking', 'jogging']")
        print("="*60 + "\n")
        
        # Process with merged classes
        process_merged_classes(
            extractor=extractor,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            merge_files=args.merge,
            sequence_length=args.sequence_length,
            skip_pose_validation=args.skip_pose_validation
        )
    else:
        # Original processing (4 separate classes)
        extractor.process_dataset(
            classes=args.classes,
            max_frames=args.max_frames,
            split_data=not args.no_split,
            output_dir=args.output_dir,
            merge_files=args.merge,
            sequence_length=args.sequence_length,
            skip_pose_validation=args.skip_pose_validation
        )
    
    print("\n💡 Next steps:")
    if not args.no_split:
        print(f"   1. Features saved in {args.output_dir}/train, /val, /test")
        if args.merge:
            print(f"   2. Merged datasets: train_merged_dataset.npz, val_merged_dataset.npz, test_merged_dataset.npz")
            print(f"   3. Load in Python: data = np.load('dataset/train_merged_dataset.npz')")
        print("   4. Train model: python train.py --max_sequence_length 30")
        print("   5. Each .npy file: (num_frames, 1412)")
    else:
        print("   1. Features saved as *_hybrid.npy in source folders")
        print("   2. Train model: python train.py --max_sequence_length 30")
    
    print("\n💡 Augmentation types applied:")
    if not args.no_augmentation:
        print("   - Original: No changes")
        print("   - Speed slow: 0.8x playback speed")
        print("   - Speed fast: 1.2x playback speed")
        print("   - Crop: Random 80-100% temporal crop")
        print("   - Sample: Every 2nd frame sampling")
        print("   - Time shift: ±3–5 frames (circular)")
        print("   - Gaussian noise: σ 0.005–0.01")
        print("   - Feature dropout: 5% random zeroing")
        print("   - Temporal masking: ≤5% frame masking")
        print("   - Horizontal flip: Mirror pose landmarks (NEW)")  # NEW
    else:
        print("   - None (augmentation disabled)")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()