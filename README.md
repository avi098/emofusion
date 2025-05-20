# Emofusion

This project performs **real-time emotion detection** using **facial expressions** and **speech signals**. It leverages deep learning models trained on image and audio data to classify emotions such as happy, sad, angry, etc. It includes a user interface to interact with these models and displays detected emotions visually and/or audibly.

---

## 🔍 Features

- Real-time facial emotion detection via webcam
- Real-time speech emotion recognition from microphone
- Pre-trained `.h5` models for facial and speech input
- Integrated dashboard and logging
- SQLite database support

---

## 🛠️ Setup Instructions

### 1. Clone or Download

Download the repository and extract the speech and facial emotion detection folder.

### 2. Environment Setup

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Make sure a `.env` file exists in the root directory with necessary environment variables (if any). Sample:

```
FLASK_ENV=development
```

---

## 🚀 How to Run

### Run the full application:

```bash
python main.py
```

This will launch both facial and speech emotion detection modules.

### Alternatively, you can run individual components:

- **App UI (if Flask or Streamlit-based):**
  ```bash
  python app.py
  ```

- **Video webcam-based emotion detection:**
  ```bash
  python video.py
  ```

---

## 📦 Project Structure

```
├── app.py                  # Application launcher
├── main.py                 # Main orchestrator
├── video.py                # Webcam-based face emotion detection
├── face_emotion_model.h5   # Trained CNN for facial emotions
├── speech_emotion_model.h5 # Trained model for speech emotion
├── emofusion.db            # SQLite database
├── requirements.txt        # Python dependencies
├── .env                    # Environment config
├── app.log                 # Application log
└── readme.txt              # Original notes
```

---

## 📄 License

This project is for academic/research purposes. Please refer to individual file headers and model licenses for more details.
