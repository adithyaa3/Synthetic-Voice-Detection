import os
import sys
import numpy as np
import librosa
import joblib

print("=== SYNTHETIC VOICE DETECTOR - SVM PREDICTION ===")


def extract_features(audio_path):

    # Load first 4 seconds
    y, sr = librosa.load(
        audio_path,
        sr=22050,
        duration=4
    )

    # Extract 40 MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    # Mean of each MFCC
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    # Standard deviation of each MFCC
    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # Combine them
    features = np.concatenate(
        (mfcc_mean, mfcc_std)
    )

    return features


def predict_audio(audio_path):

    try:

        BASE_PATH = os.path.dirname(
            os.path.dirname(__file__)
        )

        model_path = os.path.join(
            BASE_PATH,
            "models",
            "voice_detector_svm.pkl"
        )

        scaler_path = os.path.join(
            BASE_PATH,
            "models",
            "scaler.pkl"
        )

        # Load trained SVM
        model = joblib.load(
            model_path
        )

        # Load scaler
        scaler = joblib.load(
            scaler_path
        )

        print("\nProcessing audio...")

        # Extract MFCC features
        features = extract_features(
            audio_path
        )

        # Convert to shape (1, 80)
        features = features.reshape(
            1,
            -1
        )

        # Apply same scaling used during training
        features_scaled = scaler.transform(
            features
        )

        # Prediction
        prediction = model.predict(
            features_scaled
        )[0]

        # Probability
        probabilities = model.predict_proba(
            features_scaled
        )[0]

        confidence = probabilities[
            prediction
        ]

        if prediction == 1:

            result = "AI-GENERATED"

        else:

            result = "HUMAN"

        print("\n----------------------------")

        print(f"RESULT: {result}")

        print(
            f"Confidence: {confidence:.2%}"
        )

        print(
            f"File: {os.path.basename(audio_path)}"
        )

        print("----------------------------")

        return result, confidence

    except Exception as e:

        print(
            f"Error processing audio: {e}"
        )

        return None, None


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) > 1:

        audio_file = sys.argv[1]

    else:

        audio_file = input(
            "Enter audio file path (MP3/WAV): "
        ).strip().strip('"')

    if os.path.exists(audio_file):

        predict_audio(
            audio_file
        )

    else:

        print(
            "File not found! Please check the path."
        )