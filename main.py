# main.py - CLI interface
import os
from utils.audio_processing import convert_mp3_to_wav, record_audio
from model_utils import emotion_predictor

def display_results(result):
    """Display prediction results in a formatted way"""
    print("\n" + "="*50)
    print("🎭 EMOTION DETECTION RESULTS")
    print("="*50)
    print(f"🎯 Predicted Emotion: {result['emotion'].upper()}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print("-"*50)
    
    print("🏆 Top Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"  {i}. {pred['emotion']:<12} {pred['confidence']:.2%}")
    print("="*50)

def detect_from_mp3():
    """Detect emotion from MP3 file"""
    mp3_path = input("🎵 Enter path to MP3 file: ").strip()
    
    if not os.path.exists(mp3_path):
        print("❌ File not found!")
        return
    
    print("🔄 Converting MP3 to WAV...")
    wav_path = "converted_temp.wav"
    
    try:
        convert_mp3_to_wav(mp3_path, wav_path)
        result, error = emotion_predictor.predict_emotion(wav_path)
        
        if error:
            print(f"❌ {error}")
        else:
            display_results(result)
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(wav_path):
            os.remove(wav_path)

def detect_from_microphone():
    """Detect emotion from live microphone recording"""
    print("🎤 Live microphone recording")
    print("📝 Recording will last for 3 seconds...")
    
    wav_path = "live_recording.wav"
    
    try:
        record_audio(filename=wav_path, duration=3)
        
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            result, error = emotion_predictor.predict_emotion(wav_path)
            
            if error:
                print(f"❌ {error}")
            else:
                display_results(result)
        else:
            print("❌ Recording failed or file is empty.")
            
    except Exception as e:
        print(f"❌ Recording error: {e}")
    finally:
        # Clean up recording file
        if os.path.exists(wav_path):
            os.remove(wav_path)

def detect_from_wav():
    """Detect emotion from WAV file"""
    wav_path = input("🎵 Enter path to WAV file: ").strip()
    
    if not os.path.exists(wav_path):
        print("❌ File not found!")
        return
    
    if not wav_path.lower().endswith('.wav'):
        print("❌ Please provide a WAV file.")
        return
    
    result, error = emotion_predictor.predict_emotion(wav_path)
    
    if error:
        print(f"❌ {error}")
    else:
        display_results(result)

def check_model_status():
    """Check if model is available"""
    if emotion_predictor.is_ready():
        print("✅ Model Status: READY")
        print(f"✅ Available emotions: {', '.join(emotion_predictor.get_available_emotions())}")
        return True
    else:
        success, message = emotion_predictor.load_model()
        if success:
            print("✅ Model Status: READY")
            print(f"✅ Available emotions: {', '.join(emotion_predictor.get_available_emotions())}")
            return True
        else:
            print("❌ Model Status: NOT READY")
            print(f"💡 {message}")
            return False

def main_menu():
    """Display main menu"""
    print("\n" + "🎭" * 25)
    print("🎭        SPEECH EMOTION DETECTION        🎭")
    print("🎭" * 25)
    print("\nPlease choose an option:")
    print("1. 🎵 Detect emotion from MP3 file")
    print("2. 🎤 Detect emotion from live recording")
    print("3. 🎵 Detect emotion from WAV file")
    print("4. 📊 Check model status")
    print("5. 🚪 Exit")
    
    return input("\nEnter your choice (1-5): ").strip()

if __name__ == "__main__":
    # Check if model exists
    model_ready = check_model_status()
    
    if not model_ready:
        print("\n💡 You need to train the model first:")
        print("   Run: python train.py")
        exit(1)
    
    while True:
        choice = main_menu()
        
        if choice == "1":
            detect_from_mp3()
        elif choice == "2":
            detect_from_microphone()
        elif choice == "3":
            detect_from_wav()
        elif choice == "4":
            check_model_status()
        elif choice == "5":
            print("👋 Thank you for using Speech Emotion Detection!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        
        input("\nPress Enter to continue...")