import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv

# General categories to ignore (we want specific outputs)
GENERAL_LABELS = [
    "Animal",
    "Domestic animals, pets",
    "Livestock, farm animals, working animals",
    "Wild animals",
    "Inside, small room",
    "Outside, urban or rural",
    "Music",
    "Noise",
    "Speech",
]

def load_audio(file_path):
    """Load audio file"""
    waveform, sr = librosa.load(file_path, sr=None)
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    return waveform

def get_class_hierarchy():
    """Returns dictionary mapping display_name to hierarchy levels."""
    class_map_path = hub.load("https://tfhub.dev/google/yamnet/1").class_map_path().numpy().decode("utf-8")
    hierarchy = {}
    with open(class_map_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            display_name = row[2].strip().replace('"', '')
            hierarchy[display_name] = [part.strip() for part in display_name.split(",")]
    return hierarchy

def recognize_sound(audio_file_path, top_n=10):
    print("\nLoading YAMNet model...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    print("Model loaded.\n")

    waveform = load_audio(audio_file_path)

    # Run YAMNet
    scores, embeddings, spectrogram = yamnet(waveform)
    scores = scores.numpy()

    # Load class hierarchy
    class_hierarchy = get_class_hierarchy()
    class_names = list(class_hierarchy.keys())

    mean_scores = scores.mean(axis=0)
    top_indices = mean_scores.argsort()[-top_n:][::-1]

    # Print full list
    print("Top detected sounds:")
    for i in top_indices:
        label = class_names[i]
        confidence = mean_scores[i]
        print(f"- {label} ({confidence:.3f})")

    #  ignore general categories
    leaf_candidates = []
    for i in top_indices:
        label = class_names[i].strip()
        # Skip general categories
        if label not in GENERAL_LABELS:
            # Take the last part after comma if present
            leaf = label.split(",")[-1].strip()
            leaf_candidates.append((leaf, mean_scores[i]))

    # Pick the leaf node with highest probability
    if leaf_candidates:
        most_specific_label = max(leaf_candidates, key=lambda x: x[1])[0]
    else:
        # fallback: top label
        most_specific_label = class_names[top_indices[0]].strip()

    print("\nMost probable sound detected:")
    print(most_specific_label)

def main():
    audio_path = input("Enter path to audio file: ").strip()
    recognize_sound(audio_path)

if __name__ == "__main__":
    main()
