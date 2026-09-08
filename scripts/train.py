import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== SYNTHETIC VOICE DETECTOR - SVM TRAINING ===")

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
HUMAN_PATH = os.path.join(BASE_PATH, "human")
AI_PATH = os.path.join(BASE_PATH, "ai")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)


def extract_features(audio_path):
    """
    Extract MFCC features from an audio file.
    Returns one fixed-length feature vector.
    """

    # Load first 4 seconds of audio at 22050 Hz
    y, sr = librosa.load(
        audio_path,
        sr=22050,
        duration=4
    )

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    # Each MFCC changes over time.
    # Taking mean gives one value per coefficient.
    mfcc_mean = np.mean(mfcc, axis=1)

    # Standard deviation gives additional information
    mfcc_std = np.std(mfcc, axis=1)

    # Combine mean + std
    # 40 + 40 = 80 features
    features = np.concatenate((mfcc_mean, mfcc_std))

    return features


def load_dataset():

    print("Loading audio files...")

    features = []
    labels = []

    # ---------------- HUMAN AUDIO ----------------
    for file in os.listdir(HUMAN_PATH):

        if file.lower().endswith((".wav", ".mp3")):

            try:
                audio_path = os.path.join(HUMAN_PATH, file)

                feature_vector = extract_features(audio_path)

                features.append(feature_vector)

                # Human = 0
                labels.append(0)

                print(f"✓ Loaded Human: {file}")

            except Exception as e:
                print(f"✗ Error loading {file}: {e}")

    # ---------------- AI AUDIO ----------------
    for file in os.listdir(AI_PATH):

        if file.lower().endswith((".wav", ".mp3")):

            try:
                audio_path = os.path.join(AI_PATH, file)

                feature_vector = extract_features(audio_path)

                features.append(feature_vector)

                # AI = 1
                labels.append(1)

                print(f"✓ Loaded AI: {file}")

            except Exception as e:
                print(f"✗ Error loading {file}: {e}")

    return np.array(features), np.array(labels)


# ------------------------------------------------
# STEP 1 - LOAD DATA
# ------------------------------------------------

print("\nStep 1: Extracting MFCC features...")

X, y = load_dataset()

print("\nDataset shape:", X.shape)
print("Human samples:", np.sum(y == 0))
print("AI samples:", np.sum(y == 1))


# ------------------------------------------------
# STEP 2 - TRAIN TEST SPLIT
# ------------------------------------------------

print("\nStep 2: Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# STEP 3 - FEATURE SCALING
# ------------------------------------------------

print("\nStep 3: Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------
# STEP 4 - BUILD SVM MODEL
# ------------------------------------------------

print("\nStep 4: Building SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)


# ------------------------------------------------
# STEP 5 - TRAIN MODEL
# ------------------------------------------------

print("\nStep 5: Training SVM...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ------------------------------------------------
# STEP 6 - TEST MODEL
# ------------------------------------------------

print("\nStep 6: Evaluating model...")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Human", "AI"]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ------------------------------------------------
# STEP 7 - SAVE MODEL
# ------------------------------------------------

print("\nStep 7: Saving model...")

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

joblib.dump(
    model,
    os.path.join(MODEL_PATH, "voice_detector_svm.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_PATH, "scaler.pkl")
)

print("Model saved: models/voice_detector_svm.pkl")
print("Scaler saved: models/scaler.pkl")

print("\n=== TRAINING COMPLETED ===")