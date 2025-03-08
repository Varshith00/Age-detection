import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import sounddevice as id
import wavio

# Update the paths below with the actual paths where you have extracted the files
metadata_path = r'C:\Users\Varshith\Documents\voice samples\cv-corpus-18.0-delta-2024-06-14\en\validated.tsv'
audio_dir = r'C:\Users\Varshith\Documents\voice samples\cv-corpus-18.0-delta-2024-06-14\en\clips'

# Load the metadata file
metadata_df = pd.read_csv(metadata_path, delimiter='\t', encoding='utf-8')

# Ensure the correct path to audio files
metadata_df['file_path'] = metadata_df['path'].apply(lambda x: os.path.join(audio_dir, x))

# Map age ranges to approximate numerical values
age_mapping = {
    'teens': 15,
    'twenties': 25,
    'thirties': 35,
    'fourties': 45,
    'fifties': 55,
    'sixties': 65,
    'seventies': 75,
    'eighties': 85,
    'nineties': 95,
}

# Convert age labels to numerical values
metadata_df['age_numeric'] = metadata_df['age'].map(age_mapping)

# Filter out rows with NaN age values
metadata_df = metadata_df.dropna(subset=['age_numeric'])

# Extract the necessary columns
file_paths = metadata_df['file_path'].tolist()
age_labels = metadata_df['age_numeric'].tolist()

# Function to load an audio file
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Function to apply noise reduction to the audio signal
def apply_noise_reduction(y, sr):
    try:
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        return reduced_noise
    except Exception as e:
        print(f"Error reducing noise for sample with sr={sr}: {e}")
        return y

# Function to extract features from the audio signal
def extract_features(y, sr):
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract pitch features
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0])
        
        # Combine features into a single array
        features = np.concatenate((mfccs_mean, [pitch_mean]))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to process the audio and extract features
def process_audio(file_path):
    y, sr = load_audio(file_path)
    if y is not None and sr is not None:
        y_denoised = apply_noise_reduction(y, sr)
        features = extract_features(y_denoised, sr)
        return features
    return None

# Function to prepare the dataset by extracting features and labels
def prepare_dataset(file_paths, age_labels):
    features_list = []
    valid_labels = []
    for file_path, label in zip(file_paths, age_labels):
        features = process_audio(file_path)
        if features is not None:
            features_list.append(features)
            valid_labels.append(label)
    X = np.array(features_list)
    y = np.array(valid_labels)
    return X, y

# Prepare the dataset
X, y = prepare_dataset(file_paths, age_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f} years')

# Function to plot the waveform of an audio signal
def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.show()

# Function to record audio from the microphone
def record_audio(filename, duration, sr=16000):
    print("Recording...")
    myrecording = id.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    id.wait()  # Wait until recording is finished
    wavio.write(filename, myrecording, sr, sampwidth=2)
    print("Recording finished")

# Function to predict age from an audio file
def predict_age(file_path):
    features = process_audio(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        age_prediction = model.predict(features)
        return age_prediction[0]
    return None

# Record a new audio sample
new_audio_path = "new_sample.wav"
record_duration = 5  # seconds
record_audio(new_audio_path, record_duration)

# Predict the age from the new audio sample
predicted_age = predict_age(new_audio_path)
if predicted_age is not None:
    print(f"Predicted Age: {predicted_age:.2f} years")

    # Load the new audio file and plot the waveforms
    y, sr = load_audio(new_audio_path)
    if y is not None and sr is not None:
        plot_waveform(y, sr, 'Original Audio')

        # Apply noise reduction and plot the denoised audio
        y_denoised = apply_noise_reduction(y, sr)
        plot_waveform(y_denoised, sr, 'Denoised Audio')
else:
    print("Failed to predict age from the audio sample")
