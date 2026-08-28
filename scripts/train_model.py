import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

print("=== SYNTHETIC VOICE DETECTOR - TRAINING ===")

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
HUMAN_PATH = os.path.join(BASE_PATH, 'human')
AI_PATH = os.path.join(BASE_PATH, 'ai')

def load_and_process_audio():
    
    print("Loading audio files...")

    features = []
    labels = []

    for file in os.listdir(HUMAN_PATH):
        if file.endswith(('.mp3', '.wav')):
            try:
                audio_path = os.path.join(HUMAN_PATH, file)
                y, sr = librosa.load(audio_path, sr=22050, duration=4)

                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                if log_mel_spec.shape[1] < 862:
                    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 862 - log_mel_spec.shape[1])))
                else:
                    log_mel_spec = log_mel_spec[:, :862]

                features.append(log_mel_spec)
                labels.append(0)  # 0 for human
                print(f"✓ Loaded human: {file}")

            except Exception as e:
                print(f"✗ Error loading {file}: {e}")

    for file in os.listdir(AI_PATH):
        if file.endswith(('.mp3', '.wav')):
            try:
                audio_path = os.path.join(AI_PATH, file)
                y, sr = librosa.load(audio_path, sr=22050, duration=4)

                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                if log_mel_spec.shape[1] < 862:
                    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 862 - log_mel_spec.shape[1])))
                else:
                    log_mel_spec = log_mel_spec[:, :862]

                features.append(log_mel_spec)
                labels.append(1)  # 1 for AI
                print(f"✓ Loaded AI: {file}")

            except Exception as e:
                print(f"✗ Error loading {file}: {e}")

    features = np.array(features)
    features = (features - np.min(features)) / (np.max(features) - np.min(features))

    return features, np.array(labels)

def build_model(input_shape):
    """Build the CNN model"""
    model = models.Sequential([
        layers.Reshape((*input_shape, 1), input_shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

print("Step 1: Loading and processing audio files...")
X, y = load_and_process_audio()

print(f"Dataset shape: {X.shape}")
print(f"Labels - Human: {np.sum(y == 0)}, AI: {np.sum(y == 1)}")

print("Step 2: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 3: Building model...")
model = build_model(X_train.shape[1:])
model.summary()

print("Step 4: Training model...")
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

print("Step 5: Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

print("Step 6: Saving model...")
os.makedirs('../models', exist_ok=True)
model.save('../models/voice_detector.h5')
print("Model saved as 'models/voice_detector.h5'")

# Optional training visualization
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

print("Training completed! Now use predict.py to test audio files.")
