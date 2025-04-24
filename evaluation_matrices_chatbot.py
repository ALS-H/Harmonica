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

def normalize_mse(mse_values):
    """Normalize MSE values to a range of 0 to 1."""
    min_mse = min(mse_values)
    max_mse = max(mse_values)
    return [(mse - min_mse) / (max_mse - min_mse) for mse in mse_values]

def predict_music_similarity(reference_file, generated_file):
    """Predict similarity between reference and generated music."""
    reference_features = extract_features(reference_file)
    generated_features = extract_features(generated_file)

    if reference_features is None or generated_features is None:
        return None

    mse = compare_features(reference_features, generated_features)
    return mse

# Paths to the reference and generated music files
reference_file = "chat_bot_music.wav"  # Use Chatbot Music as the reference file
generated_files = {
    "Energetic Music": "tot_generated_music_energetic.wav",
    "Happy Music": "got_generated_music_happy.wav",
}

# Ground truth labels (e.g., "energetic", "happy")
true_labels = ["energetic", "happy"]

# Predicted labels based on similarity
predicted_labels = []

# Store MSE values for analysis
mse_values = []

print("Evaluating generated music...")
for label, file_path in generated_files.items():
    mse = predict_music_similarity(reference_file, file_path)
    if mse is not None:
        print(f"{label}: MSE = {mse:.4f}")
        mse_values.append((label, mse))
    else:
        mse_values.append((label, None))

# Normalize MSE values
mse_only = [mse for _, mse in mse_values if mse is not None]
normalized_mse = normalize_mse(mse_only)

# Assign predicted labels based on normalized MSE
for i, (label, mse) in enumerate(mse_values):
    if mse is None:
        predicted_labels.append("unknown")
    else:
        norm_mse = normalized_mse[i]
        # Adjust thresholds dynamically
        if norm_mse < 0.3:  # Threshold for "energetic"
            predicted_labels.append("energetic")
        elif norm_mse < 0.7:  # Threshold for "happy"
            predicted_labels.append("happy")
        else:
            predicted_labels.append("calm")

# Evaluate predictions
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["energetic", "happy"])
class_report = classification_report(true_labels, predicted_labels, labels=["energetic", "happy"], zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save evaluation results
output_dir = "evaluation_results_chatbot"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/evaluation_results_chatbot.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Energetic", "Happy"],
            yticklabels=["Energetic", "Happy"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
plt.savefig(f"{output_dir}/confusion_matrix.png")

# Save MSE values for analysis
mse_df = pd.DataFrame(mse_values, columns=["Generated File", "MSE"])
mse_df.to_csv(f"{output_dir}/mse_values.csv", index=False)

# Visualize MSE distribution
plt.figure(figsize=(8, 6))
sns.barplot(x="Generated File", y="MSE", data=mse_df)
plt.title("MSE Distribution")
plt.ylabel("MSE")
plt.xlabel("Generated File")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/mse_distribution.png")

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