# model_utils.py - UPDATED FOR TF 2.20.0
import os
import numpy as np
from joblib import load

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
    
    def load_model(self, model_path="models/emotion_model.h5", 
                   encoder_path="models/label_encoder.pkl"):
        """Load the model and encoder"""
        try:
            import tensorflow as tf
            
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            if not os.path.exists(encoder_path):
                return False, f"Encoder file not found: {encoder_path}"
            
            print("🔄 Loading deep learning model...")
            self.model = tf.keras.models.load_model(model_path)
            self.label_encoder = load(encoder_path)
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            return True, "Model loaded successfully"
            
        except Exception as e:
            self.is_loaded = False
            return False, f"Error loading model: {str(e)}"
    
    def predict_emotion(self, file_path):
        """Predict emotion from audio file"""
        if not self.is_loaded:
            success, message = self.load_model()
            if not success:
                return None, message
        
        try:
            from utils.feature_extraction import extract_mel_spectrogram
            
            mel_spec = extract_mel_spectrogram(file_path)
            if mel_spec is None:
                return None, "Feature extraction failed"
            
            # Prepare for prediction
            mel_spec_processed = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Predict
            prediction = self.model.predict(mel_spec_processed, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            
            emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get top predictions
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_emotions = self.label_encoder.inverse_transform(top_3_indices)
            top_3_confidences = prediction[0][top_3_indices]
            
            top_predictions = [
                {"emotion": emotion, "confidence": float(conf)}
                for emotion, conf in zip(top_3_emotions, top_3_confidences)
            ]
            
            result = {
                "emotion": emotion,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "status": "success"
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_available_emotions(self):
        """Get list of available emotion classes"""
        if self.label_encoder is not None:
            return self.label_encoder.classes_.tolist()
        return []
    
    def is_ready(self):
        """Check if model is ready for prediction"""
        return self.is_loaded

# Global instance
emotion_predictor = EmotionPredictor()