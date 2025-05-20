import sounddevice as sd
import librosa
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('speech_emotion_model.h5')

# Define the emotions in the same order as the model outputs
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

# Function to record audio from the user
def record_audio(duration=3, fs=44100):
    print("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished
    print("Recording finished!")
    return audio.flatten()

# Function to extract MFCC features from audio input
def extract_features(audio, sample_rate=44100):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)  # Taking the mean of the MFCCs
    return mfccs

# Predict emotion from the user's recorded audio
def predict_emotion():
    # Step 1: Record audio
    audio = record_audio(duration=3)
    
    # Step 2: Extract features
    features = extract_features(audio)
    
    # Step 3: Reshape features to fit the model input
    features = np.expand_dims(features, axis=0)  # Reshape to 1 sample
    features = np.expand_dims(features, axis=2)  # Add 3rd dimension for LSTM
    
    # Step 4: Predict using the model
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    
    # Step 5: Output the predicted emotion
    predicted_emotion = emotion_labels[predicted_label]
    print(f"Predicted Emotion: {predicted_emotion}")

# Call the prediction function
predict_emotion()
