"""
Train LSTM Model untuk Action Recognition (2-Phase Training + Augmentation)
Input: dataset/train, dataset/val, dataset/test (*_hybrid.npy files)
Output: model_hybrid_lstm.h5, training history, evaluation metrics
"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

class OverfittingDetector(Callback):
    """Callback untuk mendeteksi overfitting"""
    
    def __init__(self, patience=3, delta=0.02):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.wait = 0
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        train_acc = logs.get('accuracy')
        
        # Deteksi overfitting: train_acc >> val_acc
        if train_acc - val_acc > 0.05:
            print(f"\n⚠️ Overfitting detected! Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
        
        # Monitor val_loss increase
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n⚠️ Val loss not improving for {self.patience} epochs")

class HybridLSTMTrainer:
    """Train LSTM model with hybrid features (2-Phase Training)"""
    
    def __init__(self, dataset_dir='dataset', classes=['walking', 'walking1', 'jogging', 'jogging1'],
                 enable_augmentation=True, use_merged=False):
        self.dataset_dir = Path(dataset_dir)
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.enable_augmentation = enable_augmentation
        self.use_merged = use_merged
        
        print("="*60)
        print("🎯 HYBRID LSTM TRAINER")
        print("="*60)
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Dataset mode: {'MERGED (single files)' if use_merged else 'INDIVIDUAL (.npy files)'}")
        print(f"Classes: {self.classes}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Augmentation: {'ENABLED' if enable_augmentation else 'DISABLED'}")
        if enable_augmentation and not use_merged:
            print("   Real-time training augmentations:")
            print("   - Gaussian noise (σ=0.005-0.01)")
            print("   - Feature dropout (5%)")
            print("   - Temporal masking (5%)")
            print("   - Time shift (±3-5 frames)")
            print("   Note: Pre-extracted augmentations already in train set")
        print("="*60)
    
    def augment_gaussian_noise(self, features, sigma_range=(0.005, 0.01)):
        """Add Gaussian noise to features (matching ekstrak.py)"""
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, features.shape)
        noisy_features = features + noise
        # Clip to prevent extreme values
        return np.clip(noisy_features, features.min() - 3*sigma, features.max() + 3*sigma)
    
    def augment_feature_dropout(self, features, dropout_rate=0.05):
        """Randomly drop some features (matching ekstrak.py - 5%)"""
        mask = np.random.random(features.shape[1]) > dropout_rate
        augmented = features.copy()
        augmented[:, ~mask] = 0
        return augmented
    
    def augment_temporal_masking(self, features, mask_ratio=0.05):
        """Randomly mask some time steps (matching ekstrak.py - ≤5%)"""
        seq_len = features.shape[0]
        num_mask = max(1, int(seq_len * mask_ratio))
        
        if num_mask >= seq_len - 5:  # Keep at least 5 frames
            return features
        
        mask_indices = np.random.choice(seq_len, num_mask, replace=False)
        augmented = features.copy()
        
        # Interpolate from neighbors instead of zeroing
        for idx in mask_indices:
            if idx == 0:
                augmented[idx] = features[1]
            elif idx == seq_len - 1:
                augmented[idx] = features[-2]
            else:
                augmented[idx] = (features[idx-1] + features[idx+1]) / 2
        
        return augmented
    
    def augment_time_shift(self, features, shift_range=(-5, 5)):
        """Shift sequence in time (matching ekstrak.py - ±3-5 frames)"""
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        
        if shift == 0 or len(features) < 10:
            return features
        
        if shift > 0:
            # Shift forward: pad at beginning, remove from end
            padding = np.repeat(features[:1], shift, axis=0)
            shifted = np.vstack([padding, features[:-shift]])
        else:
            # Shift backward: remove from beginning, pad at end
            shift = abs(shift)
            padding = np.repeat(features[-1:], shift, axis=0)
            shifted = np.vstack([features[shift:], padding])
        
        return shifted
    
    def apply_augmentation(self, features):
        """Apply random augmentations to features (real-time during training)"""
        if not self.enable_augmentation:
            return features
        
        # Apply augmentations with conservative probabilities
        # (since dataset already has pre-extracted augmentations)
        if np.random.random() < 0.3:  # 30% chance
            features = self.augment_gaussian_noise(features, sigma_range=(0.005, 0.01))
        
        if np.random.random() < 0.2:  # 20% chance
            features = self.augment_feature_dropout(features, dropout_rate=0.05)
        
        if np.random.random() < 0.2:  # 20% chance
            features = self.augment_temporal_masking(features, mask_ratio=0.05)
        
        if np.random.random() < 0.3:  # 30% chance
            features = self.augment_time_shift(features, shift_range=(-5, 5))
        
        return features
    
    def load_merged_data(self, split='train'):
        """Load data from merged .npz file"""
        merged_file = self.dataset_dir / f"{split}_merged_dataset.npz"
        
        if not merged_file.exists():
            print(f"❌ Merged file not found: {merged_file}")
            print(f"   Run: python ekstrak.py --merge")
            return np.array([]), np.array([])
        
        print(f"\n📂 Loading {split} data from merged file...")
        print(f"   File: {merged_file}")
        
        try:
            data = np.load(merged_file)
            X = data['X']
            y = data['y']
            classes = data['classes']
            
            # Validate classes
            if not np.array_equal(classes, self.classes):
                print(f"   ⚠️ Class mismatch!")
                print(f"      Expected: {self.classes}")
                print(f"      Found: {list(classes)}")
                print(f"   Updating class mapping...")
                self.classes = list(classes)
                self.num_classes = len(classes)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
                self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            print(f"   ✅ Loaded: X shape = {X.shape}, y shape = {y.shape}")
            print(f"   Classes: {list(classes)}")
            print(f"   File size: {merged_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading merged file: {e}")
            return np.array([]), np.array([])
    
    def load_data(self, split='train', max_sequence_length=100):
        """Load data - auto-detect merged or individual files"""
        
        # Check if merged file exists
        merged_file = self.dataset_dir / f"{split}_merged_dataset.npz"
        
        if self.use_merged or merged_file.exists():
            return self.load_merged_data(split)
        else:
            return self._load_individual_data(split, max_sequence_length)
    
    def _load_individual_data(self, split='train', max_sequence_length=100):
        """Load data from individual .npy files (original method)"""
        X = []
        y = []
        
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            print(f"⚠️ {split} directory not found!")
            return np.array([]), np.array([])
        
        print(f"\n📂 Loading {split} data from individual files...")
        
        # Augmentation only for training set
        apply_aug = self.enable_augmentation and split == 'train'
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                print(f"   ⚠️ {class_name} not found in {split}")
                continue
            
            npy_files = list(class_dir.glob('*_hybrid*.npy'))
            
            print(f"   {class_name}: {len(npy_files)} samples", end='')
            
            original_count = len(npy_files)
            augmented_count = 0
            
            for npy_file in npy_files:
                try:
                    features = np.load(npy_file)
                    
                    # Limit sequence length
                    if len(features) > max_sequence_length:
                        features = features[:max_sequence_length]
                    
                    # Original sample
                    X.append(features)
                    y.append(self.class_to_idx[class_name])
                    
                    # Augmented samples (only for training)
                    if apply_aug and not npy_file.stem.endswith(('_speed_slow', '_speed_fast', '_crop', 
                                                                  '_sample', '_time_shift', '_gaussian_noise',
                                                                  '_feature_dropout', '_temporal_masking')):
                        # Only augment original files (not already augmented)
                        aug_features = self.apply_augmentation(features.copy())
                        X.append(aug_features)
                        y.append(self.class_to_idx[class_name])
                        augmented_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error loading {npy_file.name}: {e}")
            
            if apply_aug:
                print(f" → {original_count + augmented_count} (+ {augmented_count} real-time aug)")
            else:
                print()
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # Pad sequences to same length
        X = pad_sequences(X, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
        y = np.array(y)
        
        print(f"   ✅ Loaded: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train, X_val, X_test=None, scaler_path='scaler.pkl'):
        """Scale features using StandardScaler with validation"""
        print("\n📊 Scaling features...")
        
        if len(X_train) == 0:
            print("❌ No training data to fit scaler!")
            return X_train, X_val, X_test, None
        
        # Reshape for scaling: (samples, seq_len, features) → (samples*seq_len, features)
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        
        try:
            scaler.fit(X_train_reshaped)
        except Exception as e:
            print(f"❌ Error fitting scaler: {e}")
            return X_train, X_val, X_test, None
        
        # Transform all datasets
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, seq_len, n_features)
        
        if len(X_val) > 0:
            n_val = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_val, seq_len, n_features)
        else:
            X_val_scaled = X_val
        
        if X_test is not None and len(X_test) > 0:
            n_test = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test, seq_len, n_features)
        else:
            X_test_scaled = X_test
        
        # Save scaler
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"   ✅ Features scaled (StandardScaler)")
            print(f"   ✅ Scaler saved: {scaler_path}")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save scaler: {e}")
        
        # Show statistics
        print(f"   Mean: {scaler.mean_[:5]}... (first 5 features)")
        print(f"   Std: {scaler.scale_[:5]}... (first 5 features)")
        
        # Validate scaling
        scaled_mean = X_train_scaled.mean()
        scaled_std = X_train_scaled.std()
        print(f"   Validation: Scaled mean ≈ {scaled_mean:.4f}, Scaled std ≈ {scaled_std:.4f}")
        
        if abs(scaled_mean) > 0.1 or abs(scaled_std - 1.0) > 0.2:
            print(f"   ⚠️ Warning: Scaling may not be optimal!")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def build_model(self, input_shape, lstm_units=[64, 32], dropout=0.5, learning_rate=0.001,  # CHANGED: 0.7 → 0.5
                   use_label_smoothing=True):
        """Build LSTM model with BALANCED regularization (sweet spot)"""
        print("\n🏗️ Building LSTM model (Balanced Regularization)...")
        
        # Label smoothing loss
        if use_label_smoothing:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            print("   ✅ Label smoothing: 0.1 (prevents overconfidence)")
        else:
            loss_fn = 'categorical_crossentropy'
        
        model = Sequential([
            # First LSTM layer with BALANCED regularization
            Bidirectional(LSTM(lstm_units[0], return_sequences=True, 
                             kernel_regularizer=tf.keras.regularizers.l2(0.005),  # CHANGED: 0.015 → 0.005
                             recurrent_regularizer=tf.keras.regularizers.l2(0.005),  # CHANGED: 0.015 → 0.005
                             dropout=0.2,  # CHANGED: 0.4 → 0.2
                             recurrent_dropout=0.2),  # CHANGED: 0.4 → 0.2
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),  # 0.5 instead of 0.7
            
            # Second LSTM layer with BALANCED regularization
            Bidirectional(LSTM(lstm_units[1], return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.005),  # CHANGED
                             recurrent_regularizer=tf.keras.regularizers.l2(0.005),  # CHANGED
                             dropout=0.2,  # CHANGED
                             recurrent_dropout=0.2)),  # CHANGED
            BatchNormalization(),
            Dropout(dropout),
            
            # Dense layer with BALANCED L2
            Dense(32, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # CHANGED: 0.02 → 0.01
            Dropout(dropout * 0.5),  # 0.25 dropout
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        total_params = model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        print(f"⚖️ BALANCED Anti-Overfitting Strategies (Sweet Spot):")
        print(f"   - Dropout rate: {dropout} (moderate)")
        print(f"   - L2 regularization: 0.005 (balanced)")
        print(f"   - LSTM internal dropout: 0.2 (light)")
        print(f"   - Recurrent dropout: 0.2 (light)")
        print(f"   - Label smoothing: 0.1 (prevents overconfidence)")
        print(f"   - Gradient clipping: 1.0 (stabilizes training)")
        print(f"   - Batch normalization: Enabled")
        print(f"\n💡 Strategy: Moderate regularization for better learning")
        
        return model
    
    def build_lightweight_model(self, input_shape, dropout=0.5, learning_rate=0.001):
        """Build lightweight LSTM model with fewer parameters"""
        print("\n🏗️ Building Lightweight LSTM model...")
        
        model = Sequential([
            # Single LSTM layer with smaller units
            Bidirectional(LSTM(32, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),
            
            # Single dense layer
            Dense(16, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(dropout * 0.5),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        # Count parameters
        total_params = model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        
        return model
    
    def mixup_augmentation(self, X, y, alpha=0.2):
        """
        Mixup augmentation: mix two samples with random ratio
        Helps prevent overfitting by creating "in-between" samples
        
        Args:
            X: Input features (batch_size, seq_len, features)
            y: One-hot labels (batch_size, num_classes)
            alpha: Mixup strength (0.2 = conservative)
        
        Returns:
            Mixed X and y
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        
        return mixed_X, mixed_y
    
    def train(self, epochs=50, batch_size=16, max_sequence_length=30,
              model_name='model_hybrid_lstm.h5', two_phase=True, lightweight=False,
              scaler_path='scaler.pkl'):
        """Train model with 2-phase approach and feature scaling"""
        
        # Load data (auto-detect merged or individual)
        X_train, y_train = self.load_data('train', max_sequence_length)
        X_val, y_val = self.load_data('val', max_sequence_length)
        X_test, y_test = self.load_data('test', max_sequence_length)
        
        if len(X_train) == 0 or len(X_val) == 0:
            print("❌ Not enough data to train!")
            return None, None
        
        # Validate data shapes
        print(f"\n📊 Dataset Statistics:")
        print(f"   Train: {X_train.shape} ({len(np.unique(y_train))} classes)")
        print(f"   Val: {X_val.shape} ({len(np.unique(y_val))} classes)")
        if len(X_test) > 0:
            print(f"   Test: {X_test.shape} ({len(np.unique(y_test))} classes)")
        
        # CRITICAL: Validate data split ratios
        split_info = self.validate_data_split(X_train, X_val, X_test, y_train, y_val, y_test)
        
        if not split_info['is_valid']:
            print(f"\n⚠️ WARNING: Dataset split has issues")
            print(f"   However, training can continue...")
            
            import sys
            if '--force' not in sys.argv:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return None, None
        else:
            if split_info.get('is_augmented'):
                print(f"\n✅ Dataset validation PASSED (augmented train set)")
                print(f"   Original videos: ~{split_info.get('original_total', 'unknown')}")
                print(f"   Augmentation factor: ~{split_info.get('aug_factor', 0):.1f}x")
            else:
                print(f"\n✅ Dataset validation PASSED")
        
        # CRITICAL: Scale features
        X_train, X_val, X_test, scaler = self.scale_features(
            X_train, X_val, X_test, scaler_path
        )
        
        if scaler is None:
            print("❌ Scaling failed! Cannot proceed with training.")
            return None, None
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        print(f"\n🏗️ Model Configuration:")
        print(f"   Input shape: {input_shape}")
        print(f"   Output classes: {self.num_classes}")
        print(f"   Training strategy: {'2-Phase' if two_phase else 'Single Phase'}")
        print(f"   Model type: {'Lightweight' if lightweight else 'Standard (Anti-Overfitting)'}")
        
        if two_phase:
            return self._train_two_phase(
                X_train, y_train_cat, X_val, y_val_cat,
                input_shape, epochs, batch_size, model_name, lightweight
            )
        else:
            if lightweight:
                model = self.build_lightweight_model(input_shape)
            else:
                model = self.build_model(input_shape)
            return self._train_single_phase(
                model, X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, model_name
            )
    
    def evaluate(self, model, max_sequence_length=100, scaler_path='scaler.pkl'):
        """Evaluate model on test set with scaling"""
        
        print("\n📊 Evaluating model on test set...")
        
        X_test, y_test = self.load_data('test', max_sequence_length)
        
        if len(X_test) == 0:
            print("❌ No test data found!")
            return
        
        # Load scaler with validation
        print("📥 Loading scaler for test data...")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"   ✅ Scaler loaded from {scaler_path}")
        except FileNotFoundError:
            print(f"   ❌ Scaler not found: {scaler_path}")
            print(f"   ⚠️ Proceeding without scaling - results may be inaccurate!")
            X_test_scaled = X_test
        except Exception as e:
            print(f"   ❌ Error loading scaler: {e}")
            print(f"   ⚠️ Proceeding without scaling - results may be inaccurate!")
            X_test_scaled = X_test
        else:
            # Scale test features
            n_test, seq_len, n_features = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test, seq_len, n_features)
            print(f"   ✅ Test data scaled")
        
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Per-class accuracy
        print("\n📊 Per-Class Performance:")
        for cls_idx in range(self.num_classes):
            cls_name = self.idx_to_class[cls_idx]
            cls_mask = (y_test == cls_idx)
            if cls_mask.sum() > 0:
                cls_acc = (y_pred_classes[cls_mask] == cls_idx).mean()
                print(f"   {cls_name}: {cls_acc:.4f} ({cls_acc*100:.2f}%)")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.classes, digits=4))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes)
    
    def _train_two_phase(self, X_train, y_train_cat, X_val, y_val_cat,
                         input_shape, total_epochs, batch_size, model_name, lightweight=False):
        """2-Phase Training with MIXUP and improved Phase 2"""
        
        print("\n" + "="*60)
        print("🎯 TWO-PHASE TRAINING (Balanced + Mixup)")
        if lightweight:
            print("   (Lightweight Model)")
        print("="*60)
        
        # ===== PHASE 1: Initial Training with MIXUP =====
        print("\n📍 PHASE 1: Initial Training with Mixup Augmentation")
        print("   - Mixup alpha: 0.2 (conservative)")
        print("   - Label smoothing: 0.1")
        print("   - Learning rate: 0.001")
        print("   - Early stopping patience: 10")
        print("   - ReduceLROnPlateau: patience 5")
        print("-"*60)
        
        phase1_epochs = min(30, total_epochs)
        
        if lightweight:
            model = self.build_lightweight_model(input_shape, learning_rate=0.001)
        else:
            model = self.build_model(input_shape, learning_rate=0.001, use_label_smoothing=True)
        
        # Custom training loop with Mixup
        best_val_acc = 0
        patience_counter = 0
        patience = 10
        
        history_phase1 = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(phase1_epochs):
            print(f"\nEpoch {epoch+1}/{phase1_epochs}")
            
            # Training with Mixup
            num_batches = int(np.ceil(len(X_train) / batch_size))
            epoch_loss = []
            epoch_acc = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train_cat[start_idx:end_idx]
                
                # Apply Mixup with 50% probability
                if np.random.random() < 0.5:
                    X_batch, y_batch = self.mixup_augmentation(X_batch, y_batch, alpha=0.2)
                
                # Train on batch
                metrics = model.train_on_batch(X_batch, y_batch)
                epoch_loss.append(metrics[0])
                epoch_acc.append(metrics[1])
            
            # Validation
            val_metrics = model.evaluate(X_val, y_val_cat, verbose=0)
            
            # Record history
            history_phase1['loss'].append(np.mean(epoch_loss))
            history_phase1['accuracy'].append(np.mean(epoch_acc))
            history_phase1['val_loss'].append(val_metrics[0])
            history_phase1['val_accuracy'].append(val_metrics[1])
            
            print(f"loss: {np.mean(epoch_loss):.4f} - accuracy: {np.mean(epoch_acc):.4f} - "
                  f"val_loss: {val_metrics[0]:.4f} - val_accuracy: {val_metrics[1]:.4f}")
            
            # Early stopping
            if val_metrics[1] > best_val_acc:
                best_val_acc = val_metrics[1]
                patience_counter = 0
                model.save('phase1_best.h5')
                print(f"✅ Saved best model (val_acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best phase 1 model (FIXED: recompile after load)
        print("\n📥 Loading best Phase 1 model...")
        try:
            # Load model without compilation (to avoid label smoothing issues)
            model = tf.keras.models.load_model('phase1_best.h5', compile=False)
            
            # Recompile with label smoothing
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy']
            )
            print("✅ Model loaded and recompiled successfully")
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("   Continuing with current model in memory...")
        
        phase1_loss, phase1_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\n✅ Phase 1 Complete:")
        print(f"   Best Val Loss: {phase1_loss:.4f}")
        print(f"   Best Val Accuracy: {phase1_acc:.4f}")
        
        # FIXED: Check early stopping without .history
        if len(history_phase1['loss']) < 10:  # CHANGED: Removed .history
            print(f"\n⚠️ WARNING: Phase 1 stopped very early (epoch {len(history_phase1['loss'])})")
            print(f"   This suggests severe overfitting!")
            print(f"   Recommendations:")
            print(f"   1. Increase dataset size")
            print(f"   2. Current regularization is balanced - don't increase further")
        
        # ===== PHASE 2: Fine-tuning =====
        print("\n" + "="*60)
        print("📍 PHASE 2: Fine-tuning")
        print("   - Learning rate: 0.0005")
        print("   - Patience: 7")
        print("   - Min delta: 0.0005")
        print("-"*60)
        
        # CRITICAL: Update learning rate by rebuilding optimizer
        print("📍 Updating optimizer with new learning rate...")
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
        
        model.compile(
            optimizer=new_optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )
        print(f"✅ Learning rate updated: 0.0005")
        
        phase2_epochs = total_epochs - len(history_phase1['loss'])
        
        if phase2_epochs <= 0:
            print("\n⚠️ No epochs left for Phase 2, using Phase 1 model")
            history_obj = type('History', (), {})()
            history_obj.history = history_phase1
            self.plot_history(history_obj)
            return model, history_obj
        
        phase2_callbacks = [
            ModelCheckpoint(
                model_name,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1,
                mode='min',
                min_delta=0.0005
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"📍 Starting Phase 2 training from epoch {len(history_phase1['loss']) + 1}...")
        
        history_phase2 = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=phase2_epochs + len(history_phase1['loss']),
            batch_size=batch_size,
            callbacks=phase2_callbacks,
            verbose=1,
            initial_epoch=len(history_phase1['loss'])
        )
        
        # Check if phase 2 actually trained
        if len(history_phase2.history.get('loss', [])) == 0:
            print("\n⚠️ Phase 2 stopped immediately, using Phase 1 model")
            model.save(model_name)
            history_obj = type('History', (), {})()
            history_obj.history = history_phase1
            self.plot_history(history_obj)
            return model, history_obj
        
        # Combine histories
        for key in history_phase1.keys():
            if key in history_phase2.history:
                history_phase1[key].extend(history_phase2.history[key])
        
        # Convert to Keras history format
        history_obj = type('History', (), {})()
        history_obj.history = history_phase1
        
        # Evaluate final model
        final_loss, final_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\n✅ Phase 2 Complete:")
        print(f"   Final Val Loss: {final_loss:.4f}")
        print(f"   Final Val Accuracy: {final_acc:.4f}")
        
        # Compare
        print(f"\n{'='*60}")
        print("📊 PHASE COMPARISON:")
        print(f"   Phase 1: Loss={phase1_loss:.4f}, Acc={phase1_acc:.4f}")
        print(f"   Phase 2: Loss={final_loss:.4f}, Acc={final_acc:.4f}")
        
        if final_loss <= phase1_loss and final_acc >= phase1_acc - 0.01:
            print("   ✅ Phase 2 improved or maintained performance!")
        else:
            print("   ⚠️ Phase 2 performance degraded, consider using Phase 1 model")
        print("="*60)
        
        # Plot training history with phase markers
        self.plot_history(history_obj, phase1_end=len(history_phase1['loss']) - len(history_phase2.history.get('loss', [])))
        
        return model, history_obj
    
    def _train_single_phase(self, model, X_train, y_train_cat, X_val, y_val_cat,
                           epochs, batch_size, model_name):
        """Single phase training (original)"""
        
        callbacks = [
            ModelCheckpoint(
                model_name,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Training model (Single Phase)...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples: {len(X_val)}")
        print("="*60)
        
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.plot_history(history)
        
        return model, history
    
    def _combine_histories(self, hist1, hist2):
        """Combine two history objects"""
        combined = type('History', (), {})()
        combined.history = {}
        
        # Get all keys from first history
        for key in hist1.history.keys():
            if key in hist2.history:
                combined.history[key] = hist1.history[key] + hist2.history[key]
            else:
                # If key doesn't exist in hist2, just use hist1
                combined.history[key] = hist1.history[key]
        
        return combined
    
    def plot_history(self, history, phase1_end=None):
        """Plot training history with phase markers"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Accuracy
        axes[0].plot(epochs, history.history['accuracy'], label='Train', marker='o', markersize=3)
        axes[0].plot(epochs, history.history['val_accuracy'], label='Validation', marker='s', markersize=3)
        
        if phase1_end:
            axes[0].axvline(x=phase1_end, color='red', linestyle='--', label=f'Phase 1 End (Epoch {phase1_end})')
        
        axes[0].set_title('Model Accuracy (2-Phase Training)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(epochs, history.history['loss'], label='Train', marker='o', markersize=3)
        axes[1].plot(epochs, history.history['val_loss'], label='Validation', marker='s', markersize=3)
        
        if phase1_end:
            axes[1].axvline(x=phase1_end, color='red', linestyle='--', label=f'Phase 1 End (Epoch {phase1_end})')
        
        axes[1].set_title('Model Loss (2-Phase Training)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("\n✅ Training history saved: training_history.png")
        plt.show()  # Tampilkan grafik
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("✅ Confusion matrix saved: confusion_matrix.png")
        plt.show()  # Tampilkan grafik
    
    def validate_data_split(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Validate dataset split ratios and warn about imbalanced splits
        FIXED: Better augmentation detection based on actual split
        """
        total_samples = len(X_train) + len(X_val) + len(X_test)
        
        train_ratio = len(X_train) / total_samples
        val_ratio = len(X_val) / total_samples
        test_ratio = len(X_test) / total_samples
        
        print(f"\n📊 Dataset Split Validation:")
        print(f"   Total samples (after augmentation): {total_samples}")
        print(f"   Train: {len(X_train)} ({train_ratio*100:.1f}%)")
        print(f"   Val: {len(X_val)} ({val_ratio*100:.1f}%)")
        print(f"   Test: {len(X_test)} ({test_ratio*100:.1f}%)")
        
        # CRITICAL FIX: Better augmentation detection
        # If val+test ~= 20% of train (unaugmented), then train is augmented
        val_test_sum = len(X_val) + len(X_test)
        
        # Expected: val+test = 20% of total original videos
        # If train is augmented 6x: train_files = 6 * original_train
        # Then: val_test_sum / (val_test_sum / 0.2) should equal original_train
        
        if train_ratio > 0.90:  # Suspiciously high train ratio (>90%)
            print(f"\n⚠️ DETECTED: Training set appears to be AUGMENTED")
            print(f"   High train ratio ({train_ratio*100:.1f}%) suggests augmentation")
            
            # Better estimation based on val+test being 20% of original
            # original_total = val_test_sum / 0.2
            # original_train = original_total * 0.8
            original_total_estimated = val_test_sum / 0.2
            original_train_estimated = original_total_estimated * 0.8
            
            print(f"\n   📊 Estimated Original Split (before augmentation):")
            print(f"   - Original total videos: ~{int(original_total_estimated)}")
            print(f"   - Original train: ~{int(original_train_estimated)} (80%)")
            print(f"   - Val: {len(X_val)} (10%)")
            print(f"   - Test: {len(X_test)} (10%)")
            
            # Calculate augmentation factor
            aug_factor = len(X_train) / original_train_estimated
            print(f"   - Estimated augmentation factor: ~{aug_factor:.1f}x")
            
            # Verify if val+test ratio is reasonable (should be ~20% of original total)
            val_test_ratio = val_test_sum / original_total_estimated
            
            if 0.15 <= val_test_ratio <= 0.25:  # 15-25% is reasonable
                print(f"\n   ✅ Dataset split is VALID (augmented train set)")
                print(f"   This is EXPECTED when using pre-extracted augmentations")
                
                # Only warn about small absolute numbers
                if len(X_val) < 30:
                    print(f"\n   ℹ️ NOTE: Validation set is small ({len(X_val)} samples)")
                    print(f"   Each error = {100/len(X_val):.1f}% accuracy change")
                    print(f"   This is normal for datasets with <250 original videos")
                else:
                    print(f"\n   ✅ Validation set size is adequate ({len(X_val)} samples)")
                
                return {
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'is_valid': True,  # Accept augmented dataset
                    'is_augmented': True,
                    'original_total': int(original_total_estimated),
                    'aug_factor': aug_factor
                }
            else:
                print(f"\n   ⚠️ WARNING: Val+Test ratio seems off ({val_test_ratio*100:.1f}%)")
                print(f"   Expected: 20% of original videos")
        
        # Non-augmented dataset validation
        if 0.05 <= val_ratio <= 0.15 and 0.05 <= test_ratio <= 0.15:
            print(f"\n   ✅ Dataset split appears VALID (non-augmented)")
            return {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'is_valid': True,
                'is_augmented': False
            }
        
        # Small validation set warning
        if len(X_val) < 50:
            print(f"\n   ⚠️ Small validation set ({len(X_val)} samples)")
            print(f"   Each error = {100/len(X_val):.1f}% accuracy change")
            
            if len(X_val) < 20:
                print(f"   ⚠️ CRITICAL: Validation set too small!")
                print(f"   Recommendation: Need at least 20-30 validation samples")
                return {
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'is_valid': False,
                    'is_augmented': False
                }
        
        return {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'is_valid': True,  # Accept anyway with warning
            'is_augmented': False
        }
    
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM model with hybrid features (2-Phase Training)')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                       help='Dataset directory (default: dataset)')
    parser.add_argument('--classes', nargs='+', default=['walking', 'jogging'],
                       help='List of classes (default: walking, jogging)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--max_sequence_length', type=int, default=30,
                       help='Max sequence length (default: 30 for KTH real-time)')
    parser.add_argument('--model_name', type=str, default='model_hybrid_lstm.h5',
                       help='Output model name')
    parser.add_argument('--two_phase', action='store_true', default=True,
                       help='Use 2-phase training strategy (default: True)')
    parser.add_argument('--single_phase', action='store_true',
                       help='Use single phase training (disable 2-phase)')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight model (recommended for small datasets)')  # UPDATED
    parser.add_argument('--evaluate_only', type=str, default=None,
                       help='Only evaluate existing model (provide model path)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl',
                       help='Path to save scaler (default: scaler.pkl)')
    parser.add_argument('--use_merged', action='store_true',
                       help='Force use merged dataset files (.npz)')
    parser.add_argument('--validate_scaler', type=str, default=None,
                       help='Validate scaler file (provide path to scaler.pkl)')
    parser.add_argument('--force', action='store_true',  # NEW
                       help='Force training even with invalid dataset split')
    
    args = parser.parse_args()
    
    # Validate sequence length
    if args.max_sequence_length != 30:
        print(f"⚠️ WARNING: Using sequence_length={args.max_sequence_length}")
        print(f"   KTH dataset was extracted with sequence_length=30")
        print(f"   Mismatch may cause issues. Recommended: --max_sequence_length 30")
    
    # Validate scaler mode
    if args.validate_scaler:
        print(f"📊 Validating scaler: {args.validate_scaler}")
        try:
            with open(args.validate_scaler, 'rb') as f:
                scaler = pickle.load(f)
            print(f"   ✅ Scaler loaded successfully")
            print(f"   Features: {len(scaler.mean_)}")
            print(f"   Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
            print(f"   Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
            
            # Test transform
            dummy_data = np.random.randn(10, len(scaler.mean_))
            transformed = scaler.transform(dummy_data)
            print(f"   Test transform: ✅")
            print(f"   Transformed mean: {transformed.mean():.4f}")
            print(f"   Transformed std: {transformed.std():.4f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        return
    
    # Override two_phase if single_phase is specified
    if args.single_phase:
        args.two_phase = False
    
    trainer = HybridLSTMTrainer(
        dataset_dir=args.dataset_dir,
        classes=args.classes,
        enable_augmentation=not args.no_augmentation,
        use_merged=args.use_merged
    )
    
    if args.evaluate_only:
        # Load and evaluate existing model
        print(f"📥 Loading model: {args.evaluate_only}")
        try:
            model = tf.keras.models.load_model(args.evaluate_only)
            trainer.evaluate(model, args.max_sequence_length, args.scaler_path)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        # Train new model
        model, history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            model_name=args.model_name,
            two_phase=args.two_phase,
            lightweight=args.lightweight,
            scaler_path=args.scaler_path
        )
        
        if model and history:
            trainer.evaluate(model, args.max_sequence_length, args.scaler_path)
            print(f"\n✅ Training complete! Model saved: {args.model_name}")
            print(f"✅ Scaler saved: {args.scaler_path}")
        else:
            print("\n❌ Training failed!")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
