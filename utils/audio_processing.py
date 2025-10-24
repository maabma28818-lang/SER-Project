# utils/audio_processing.py
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np

def convert_mp3_to_wav(mp3_path, wav_path, target_sr=22050):
    """Convert MP3 to WAV with target sample rate"""
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(target_sr)
    audio = audio.set_channels(1)  # Mono
    audio.export(wav_path, format="wav")

def record_audio(filename="live.wav", duration=3, fs=22050):
    """Record audio with consistent settings for model"""
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    
    # Normalize the recorded audio
    y, sr = librosa.load(filename, sr=fs)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    write(filename, fs, (y * 32767).astype(np.int16))  # Save normalized
    
    print(f"Recording saved as {filename}")