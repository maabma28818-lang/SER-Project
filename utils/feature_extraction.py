# utils/feature_extraction.py
import numpy as np
import librosa
import os

def load_audio_file(file_path, sr=22050, duration=3):
    """Load audio with fixed duration for consistent spectrogram size"""
    try:
        y, sr_ret = librosa.load(file_path, sr=sr, mono=True, duration=duration)
        
        # If audio is shorter than duration, pad it
        if len(y) < sr * duration:
            y = np.pad(y, (0, int(sr * duration) - len(y)), mode='constant')
        
        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        return y, sr_ret
    except Exception as e:
        print(f"Could not load {file_path}: {e}")
        return None, None

def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, duration=3):
    """Extract 2D Mel-spectrogram for deep learning"""
    y, sr_ret = load_audio_file(file_path, sr=sr, duration=duration)
    if y is None:
        return None
    
    try:
        # Create Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr_ret, n_mels=n_mels, 
            n_fft=2048, hop_length=512
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure consistent shape (n_mels x time_frames)
        target_frames = 130  # Based on 3-second audio at 22050Hz
        if log_mel_spec.shape[1] < target_frames:
            log_mel_spec = np.pad(
                log_mel_spec, 
                ((0, 0), (0, target_frames - log_mel_spec.shape[1])), 
                mode='constant'
            )
        elif log_mel_spec.shape[1] > target_frames:
            log_mel_spec = log_mel_spec[:, :target_frames]
            
        return log_mel_spec
        
    except Exception as e:
        print(f"Mel-spectrogram failed for {file_path}: {e}")
        return None

# Keep old function for compatibility during transition
def extract_features(file_path, sr=22050, n_mfcc=40):
    """Legacy function - use extract_mel_spectrogram for deep learning"""
    return extract_mel_spectrogram(file_path, sr, n_mels=128)