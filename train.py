# train.py - UPDATED FOR TF 2.20.0
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump
import librosa

from utils.feature_extraction import extract_mel_spectrogram
from models.deep_learning_model import create_emotion_model, compile_model

# --- Emotion code mappings ---
RAVDESS_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def parse_label_from_filename(file_path):
    """Detect emotion label from filename"""
    frame = os.path.basename(file_path).lower()
    
    # RAVDESS
    if '-' in frame:
        parts = frame.split('-')
        if len(parts) >= 3 and parts[2].isdigit() and parts[2] in RAVDESS_MAP:
            return RAVDESS_MAP[parts[2]]
    
    # TESS
    for emo in ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'calm']:
        if emo in frame:
            return emo
    
    return None

def augment_audio(y, sr):
    """Apply data augmentation to audio"""
    augmented = []
    
    # Original
    augmented.append(y)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.005, y.shape)
    augmented.append(y + noise)
    
    # Pitch shifting (±2 semitones)
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except:
        # If pitch shifting fails, just use the original
        augmented.append(y)
        augmented.append(y)
    
    return augmented

def load_dataset_with_augmentation(data_dir='data', augment=True):
    """Load dataset with optional augmentation"""
    X, y = [], []
    skipped = 0
    
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith('.wav'):
                continue
                
            full_path = os.path.join(root, f)
            label = parse_label_from_filename(full_path)
            
            if not label:
                print(f"SKIP: Label not found for {full_path}")
                skipped += 1
                continue
            
            # Extract base features
            mel_spec = extract_mel_spectrogram(full_path)
            if mel_spec is None:
                print(f"SKIP: Feature extraction failed for {full_path}")
                skipped += 1
                continue
            
            X.append(mel_spec)
            y.append(label)
            
            # Data augmentation
            if augment:
                try:
                    y_audio, sr = librosa.load(full_path, sr=22050)
                    augmented_versions = augment_audio(y_audio, sr)
                    
                    for aug_audio in augmented_versions[1:]:  # Skip original
                        # Convert augmented audio to mel-spectrogram
                        mel_spec_aug = librosa.feature.melspectrogram(
                            y=aug_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
                        )
                        log_mel_spec_aug = librosa.power_to_db(mel_spec_aug, ref=np.max)
                        
                        # Ensure consistent shape
                        target_frames = 130
                        if log_mel_spec_aug.shape[1] < target_frames:
                            log_mel_spec_aug = np.pad(
                                log_mel_spec_aug, 
                                ((0, 0), (0, target_frames - log_mel_spec_aug.shape[1])), 
                                mode='constant'
                            )
                        elif log_mel_spec_aug.shape[1] > target_frames:
                            log_mel_spec_aug = log_mel_spec_aug[:, :target_frames]
                        
                        X.append(log_mel_spec_aug)
                        y.append(label)
                        
                except Exception as e:
                    print(f"Augmentation failed for {full_path}: {e}")
                    continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} samples (including augmentations), skipped {skipped} files.")
    return X, y

def prepare_data_for_training(X, y):
    """Prepare data for deep learning training"""
    # Add channel dimension for CNN
    X = X[..., np.newaxis]  # Shape: (samples, height, width, channels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    return X, y_categorical, label_encoder

if __name__ == "__main__":
    print("🎭 Starting Speech Emotion Detection Training...")
    
    # Load data with augmentation
    X, y = load_dataset_with_augmentation('data', augment=True)
    
    if len(X) == 0:
        print("❌ No data found. Make sure .wav files are inside data/ folder")
        print("💡 Download RAVDESS and TESS datasets and place them in data/ folder")
        exit(1)
    
    # Prepare data for training
    X_processed, y_processed, label_encoder = prepare_data_for_training(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.15, random_state=42, stratify=y
    )
    
    # Create and compile model
    input_shape = X_train.shape[1:]  # (height, width, channels)
    num_classes = y_processed.shape[1]
    
    print(f"📊 Input shape: {input_shape}")
    print(f"🎯 Number of classes: {num_classes}")
    print(f"📈 Classes: {label_encoder.classes_}")
    
    # Create model
    model = create_emotion_model(input_shape, num_classes)
    model = compile_model(model, learning_rate=0.001)
    
    print("🧠 Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_emotion_model.h5', 
            save_best_only=True, 
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("🚀 Training deep learning model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # Save final model and encoders
    os.makedirs('models', exist_ok=True)
    model.save('models/emotion_model.h5')
    dump(label_encoder, 'models/label_encoder.pkl')
    
    print("✅ Model and encoders saved successfully in models/ folder!")
    print("🎉 Training completed! You can now run:")
    print("   python app.py     # For web interface")
    print("   python main.py    # For CLI interface")