# app.py - Main Flask application
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from utils.audio_processing import convert_mp3_to_wav
from model_utils import emotion_predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/live_mic')
def live_mic():
    return render_template('live_mic.html')

@app.route('/audio_analysis')
def audio_analysis():
    return render_template('audio_analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle emotion prediction from uploaded audio files"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        return jsonify({'error': 'Only WAV and MP3 files are supported'}), 400
    
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Convert MP3 to WAV if necessary
        if filename.endswith('.mp3'):
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted.wav')
            convert_mp3_to_wav(temp_path, wav_path)
            result, error = emotion_predictor.predict_emotion(wav_path)
            # Clean up converted file
            if os.path.exists(wav_path):
                os.remove(wav_path)
        else:
            result, error = emotion_predictor.predict_emotion(temp_path)
        
        # Clean up uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    """Check if model is loaded and ready"""
    if emotion_predictor.is_ready():
        return jsonify({
            'status': 'ready',
            'message': 'Model loaded successfully',
            'classes': emotion_predictor.get_available_emotions()
        })
    else:
        success, message = emotion_predictor.load_model()
        if success:
            return jsonify({
                'status': 'ready',
                'message': 'Model loaded successfully',
                'classes': emotion_predictor.get_available_emotions()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            }), 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Try to load model on startup
    print("🚀 Starting Speech Emotion Detection Server...")
    success, message = emotion_predictor.load_model()
    if success:
        print("✅ Model loaded successfully!")
        print(f"🎭 Available emotions: {', '.join(emotion_predictor.get_available_emotions())}")
    else:
        print(f"❌ Model loading failed: {message}")
        print("💡 You need to train the model first. Run: python train.py")
    
    print("\n🌐 Server running on: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)