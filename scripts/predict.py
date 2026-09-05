import os
import numpy as np
import librosa
import tensorflow as tf
import sys

print("=== SYNTHETIC VOICE DETECTOR - PREDICTION ===")

def predict_audio(audio_path):
    """Predict if audio is Human or AI"""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=4)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        if log_mel_spec.shape[1] < 862:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 862 - log_mel_spec.shape[1])))
        else:
            log_mel_spec = log_mel_spec[:, :862]

        log_mel_spec = (log_mel_spec - np.min(log_mel_spec)) / (np.max(log_mel_spec) - np.min(log_mel_spec))

        input_data = np.expand_dims(log_mel_spec, axis=(0, -1))

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'voice_detector.h5')
        model = tf.keras.models.load_model(model_path)

        prediction = model.predict(input_data, verbose=0)[0][0]

        if prediction > 0.5:
            result = "AI-GENERATED"
            confidence = prediction
        else:
            result = "HUMAN"
            confidence = 1 - prediction

        print(f"\n RESULT: {result}")
        print(f" Confidence: {confidence:.2%}")
        print(f" File: {os.path.basename(audio_path)}")

        return result, confidence

    except Exception as e:
        print(f" Error processing audio: {e}")
        return None, None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Enter the path to your audio file (MP3/WAV): ").strip().strip('"')

    if os.path.exists(audio_file):
        predict_audio(audio_file)
    else:
        print(" File not found! Please check the path.")
