# test_complete_system.py - Test the entire pipeline
import numpy as np
import tensorflow as tf
import os
from joblib import load

print("🧪 Testing Complete Speech Emotion Detection System...")

# Test 1: TensorFlow
print("1. Testing TensorFlow...")
try:
    print(f"   ✅ TensorFlow {tf.__version__} - OK")
except Exception as e:
    print(f"   ❌ TensorFlow failed: {e}")

# Test 2: Model creation
print("2. Testing model creation...")
try:
    from models.deep_learning_model import create_emotion_model, compile_model
    model = create_emotion_model(input_shape=(128, 130, 1), num_classes=8)
    model = compile_model(model)
    print("   ✅ Model creation - OK")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")

# Test 3: Feature extraction
print("3. Testing feature extraction...")
try:
    from utils.feature_extraction import extract_mel_spectrogram
    print("   ✅ Feature extraction imports - OK")
except Exception as e:
    print(f"   ❌ Feature extraction failed: {e}")

# Test 4: Audio processing
print("4. Testing audio processing...")
try:
    from utils.audio_processing import convert_mp3_to_wav, record_audio
    print("   ✅ Audio processing imports - OK")
except Exception as e:
    print(f"   ❌ Audio processing failed: {e}")

# Test 5: Model utils
print("5. Testing model utilities...")
try:
    from model_utils import emotion_predictor
    print("   ✅ Model utilities - OK")
except Exception as e:
    print(f"   ❌ Model utilities failed: {e}")

print("\n🎉 System test completed!")
print("💡 Next step: Run 'python train.py' to train your model")