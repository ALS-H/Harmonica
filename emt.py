import numpy as np
import pickle

# Define emotion mapping using rule-based approach
emotion_mapping = {
    "Happy": {"tempo": (120, 200), "key": "major", "note_density": (5, 15), "velocity": (60, 127)},
    "Sad": {"tempo": (40, 90), "key": "minor", "note_density": (1, 6), "velocity": (30, 80)},
    "Calm": {"tempo": (50, 100), "key": "major", "note_density": (2, 8), "velocity": (30, 70)},
    "Energetic": {"tempo": (150, 250), "key": "major", "note_density": (10, 25), "velocity": (80, 127)},
    "Romantic": {"tempo": (50, 120), "key": "minor", "note_density": (3, 10), "velocity": (50, 100)},
    "Fearful": {"tempo": (40, 110), "key": "minor", "note_density": (4, 12), "velocity": (40, 100)},
    "Angry": {"tempo": (140, 230), "key": "minor", "note_density": (12, 30), "velocity": (90, 127)},
    "Mysterious": {"tempo": (60, 120), "key": "minor", "note_density": (5, 15), "velocity": (40, 90)}
}
# Load preprocessed MIDI features
with open("processed_midi_data.pkl", "rb") as f:
    midi_features = pickle.load(f)
def classify_emotion_rule_based(features):
    """
    Classifies the emotion of a MIDI file based on predefined rule-based mapping.
    """
    for emotion, params in emotion_mapping.items():
        tempo_range, key_type, note_range, velocity_range = (
            params["tempo"], params["key"], params["note_density"], params["velocity"]
        )

        # Convert key number to major/minor (0-11 major, 12-23 minor)
        key_category = "major" if features["key"] < 12 else "minor"

        # Check if the features match the emotion category
        if (
            tempo_range[0] <= features["tempo"] <= tempo_range[1]
            and note_range[0] <= features["note_density"] <= note_range[1]
            and velocity_range[0] <= features["avg_velocity"] <= velocity_range[1]
            and key_category == key_type
        ):
            return emotion
    
    return "Unknown"  # Default if no match

# Apply rule-based classification
midi_emotion_labels = {file: classify_emotion_rule_based(features) for file, features in midi_features.items()}
import pandas as pd

# Convert to DataFrame and save
emotion_df = pd.DataFrame.from_dict(midi_emotion_labels, orient="index", columns=["Emotion"])
emotion_df.to_csv("midi_emotion_labels.csv")

print("Rule-based emotion classification completed and saved!")
