import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# Specify dataset path
dataset_path = 'data/voice/TESS Toronto emotional speech set data/'

# Load dataset paths and labels
def load_dataset(path):
    paths, labels = [], []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.wav'):
                paths.append(os.path.join(dirname, filename))
                label = filename.split('_')[-1].split('.')[0].lower()
                label = 'pleasant_surprise' if label == 'ps' else label
                labels.append(label)
    return pd.DataFrame({'speech': paths, 'label': labels})

df = load_dataset(dataset_path)
print(f'Dataset Loaded: {len(df)} files')
print(df['label'].value_counts())

# Visualize the distribution of emotion labels
sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
plt.title("Emotion Label Distribution")
plt.show()

# Functions to visualize audio
def visualize_audio(data, sr, emotion):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Audio Analysis - {emotion.capitalize()}', fontsize=16)

    # Waveplot
    axs[0].set_title("Waveform")
    librosa.display.waveshow(data, sr=sr, ax=axs[0])

    # Spectrogram
    xdb = librosa.amplitude_to_db(abs(librosa.stft(data)))
    img = librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
    axs[1].set_title("Spectrogram")
    plt.colorbar(img, ax=axs[1])
    plt.show()

# Preview a single file per emotion
for emotion in df['label'].unique():
    file_path = df[df['label'] == emotion]['speech'].iloc[0]
    audio_data, sr = librosa.load(file_path, duration=3)
    visualize_audio(audio_data, sr, emotion)

# Extract MFCC features
def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, duration=3, offset=0.5)
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

df['mfcc'] = df['speech'].apply(extract_mfcc)

# Convert to numpy arrays
X = np.array(df['mfcc'].tolist())
y = df['label']

# Encode labels and split data
label_encoder = LabelEncoder()
y_encoded = to_categorical(label_encoder.fit_transform(y))
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape for LSTM
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

# Build the LSTM model
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and summarize the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, verbose=1)

# Plot training results
def plot_history(history, metric):
    plt.plot(history.history[metric], label=f'Train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f'{metric.capitalize()} Over Epochs')
    plt.show()

plot_history(history, 'accuracy')
plot_history(history, 'loss')

# Evaluate the model
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model
model.save('speech_emotion_model.h5')
