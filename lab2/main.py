import os
from pathlib import Path
import librosa
import pandas as pd
import numpy as np
import shutil

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def get_audio_files_path():
    lab2_tmp_path = Path(__file__).parent.parent / "lab2" / "tmp"
    lab2_tmp_path.mkdir(parents=True, exist_ok=True)
    lab2_audio_files_path = lab2_tmp_path / "audio_files"
    lab2_audio_files_path.mkdir(parents=True, exist_ok=True)
    audio_files_path = Path(__file__).parent.parent / "lab1" / "tmp" / "useful_audios"
    destination_path = Path(__file__).parent.parent / "lab2" / "tmp" / "audio_files"
    for file in audio_files_path.iterdir():
        if file.is_file():
            new_file_path = destination_path / file.name
            shutil.copy(file, new_file_path)

    return destination_path


# Define a function to extract features from a single file
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    harmonic = np.mean(librosa.effects.harmonic(y).T, axis=0)
    percussive = np.mean(librosa.effects.percussive(y).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return [
        mfccs,
        chroma,
        spectral_contrast,
        tonnetz,
        spectral_centroid,
        spectral_rolloff,
        zero_crossing_rate,
        harmonic,
        percussive,
        mel,
    ]


# Define a function to process all files and extract features
def process_files(dir_path: Path):
    # Initialize empty lists to hold the features and labels
    features, file_names = [], []

    # Iterate over all files
    for file in dir_path.iterdir():

        feature = np.hstack(extract_features(file))

        features.append(feature)
        file_names.append(file.name)

    return features, file_names


audio_files_path = destination_path = Path(__file__).parent / "tmp" / "audio_files"
# Process all files and extract features
features, file_names = process_files(audio_files_path)

# Convert the features and labels to a DataFrame
df = pd.DataFrame(features)
df["Arquivo"] = file_names

# Make 'Arquivo' the first column
df = df.set_index("Arquivo").reset_index()

# Define the pipeline
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values
        ("scaler", StandardScaler()),  # Standardize features
        (
            "selector",
            SelectKBest(score_func=f_classif, k=10),
        ),  # Select the 10 best features
    ]
)

# Apply the pipeline
df[df.columns[1:]] = pipeline.fit_transform(df[df.columns[1:]])

# Save the processed DataFrame to a CSV file
df.to_csv("processed_audio_features.csv", index=False)
