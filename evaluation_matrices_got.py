import librosa
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def extract_features(file_path, target_duration=5.0, target_sr=22050):
    """Extract Mel-Spectrogram features from a .wav file."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None
        y, sr = librosa.load(file_path, sr=target_sr)  # Load audio file with target sampling rate
        # Truncate or pad the audio to the target duration
        target_length = int(target_duration * target_sr)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=target_sr)
        return mel_spectrogram
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_features(features1, features2):
    """Compare two sets of features using Mean Squared Error (MSE)."""
    try:
        # Flatten the features for comparison
        features1_flat = features1.flatten()
        features2_flat = features2.flatten()

        # Compute MSE
        mse = np.mean((features1_flat - features2_flat) ** 2)
        return mse
    except Exception as e:
        print(f"Error comparing features: {e}")
        return None

def predict_music_similarity(reference_file, generated_file):
    """Predict similarity between reference and generated music."""
    reference_features = extract_features(reference_file)
    generated_features = extract_features(generated_file)

    if reference_features is None or generated_features is None:
        return None

    mse = compare_features(reference_features, generated_features)
    return mse

# Paths to the reference and generated music files
reference_file = "GoT_generated_music_happy.wav"  # Use GoT-generated happy music as the reference file
generated_files = {
    "Energetic Music": "tot_generated_music_energetic.wav",
    "Chatbot Music": "chat_bot_music.wav",
}

# Ground truth labels (e.g., "energetic", "calm")
true_labels = ["energetic", "calm"]

# Predicted labels based on similarity
predicted_labels = []

print("Evaluating generated music...")
for label, file_path in generated_files.items():
    mse = predict_music_similarity(reference_file, file_path)
    if mse is not None:
        print(f"{label}: MSE = {mse:.4f}")
        # Assign predicted label based on MSE (you can define thresholds for classification)
        if mse < 100:  # Example threshold for "energetic"
            predicted_labels.append("energetic")
        else:
            predicted_labels.append("calm")
    else:
        predicted_labels.append("unknown")

# Evaluate predictions
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["energetic", "calm"])
class_report = classification_report(true_labels, predicted_labels, labels=["energetic", "calm"], zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save evaluation results
output_dir = "evaluation_results_got"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/evaluation_results_got.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Energetic", "Calm"],
            yticklabels=["Energetic", "Calm"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
plt.savefig(f"{output_dir}/confusion_matrix.png")

results_df = pd.DataFrame({
    "Generated File": list(generated_files.keys()),
    "True Label": true_labels,
    "Predicted Label": predicted_labels,
    "Correct": [t == p for t, p in zip(true_labels, predicted_labels)]
})

results_df.to_csv(f"{output_dir}/prediction_examples.csv", index=False)

print("Evaluation complete!")
print(f"Results saved to {output_dir}")

print("\nExample Predictions:")
examples = results_df.sample(2).to_string(index=False)
print(examples)