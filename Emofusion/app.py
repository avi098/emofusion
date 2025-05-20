import os
import shutil
import subprocess
import sqlite3
import json
import uuid
from flask import Flask, request, jsonify, render_template_string, Response, session, redirect, url_for
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading
from queue import Queue
import google.generativeai as genai
from datetime import datetime
import requests
from pydub import AudioSegment
from pathlib import Path
import os
from gtts import gTTS
import tempfile
import base64
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import retrying  # Added for retry logic in TTS

# Download NLTK resources for sentiment analysis
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """Handles real-time facial emotion detection using OpenCV and a pre-trained model."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = self.load_emotion_model()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_history = []
        self.history_size = 5
        self.cap = None
        self.last_emotion = "Unknown"
        self.emotion_confidence = 0.0
        self.running = True
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize camera with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    logger.info("Camera initialized successfully")
                    break
            except Exception as e:
                logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if self.cap:
                    self.cap.release()
                time.sleep(1)
        
        if not self.cap or not self.cap.isOpened():
            logger.error("Failed to initialize camera after all attempts")
            self.running = False

    def load_emotion_model(self) -> Optional[Any]:
        """Load the pre-trained emotion detection model."""
        try:
            model_path = Path('face_emotion_model.h5')
            if not model_path.exists():
                logger.error("Emotion model file not found")
                return None
            return load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            return None

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion detection."""
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        return face_img

    def get_smooth_emotion(self, emotion_pred: np.ndarray) -> Tuple[str, float]:
        """Apply smoothing to emotion predictions using historical data."""
        self.emotion_history.append(emotion_pred)
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        avg_pred = np.mean(self.emotion_history, axis=0)
        emotion_idx = np.argmax(avg_pred)
        confidence = float(avg_pred[emotion_idx])
        return self.emotion_labels[emotion_idx], confidence

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Detect emotion from frame with confidence score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        max_face_area = 0
        main_face_emotion = self.last_emotion
        confidence = self.emotion_confidence
        
        for (x, y, w, h) in faces:
            face_area = w * h
            if face_area > max_face_area:
                max_face_area = face_area
                roi = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(roi)
                
                if self.emotion_model is not None:
                    emotion_pred = self.emotion_model.predict(processed_face)[0]
                    emotion_label, conf = self.get_smooth_emotion(emotion_pred)
                    main_face_emotion = emotion_label
                    confidence = conf
                    
                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{emotion_label} ({conf:.2f})"
                    cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        self.last_emotion = main_face_emotion
        self.emotion_confidence = confidence
        return frame, main_face_emotion, confidence

    def generate_frames(self):
        """Generate video frames with emotion detection."""
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break
                
            frame, _, _ = self.detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()

class SpeechEmotionDetector:
    """Handles speech emotion detection using a pre-trained model."""
    
    def __init__(self):
        self.model = self.load_model()
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
        
    def load_model(self) -> Optional[Any]:
        """Load the pre-trained speech emotion model."""
        try:
            model_path = Path('speech_emotion_model.h5')
            if not model_path.exists():
                logger.error("Speech emotion model file not found")
                return None
            return load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading speech emotion model: {e}")
            return None

    def extract_features(self, audio_path: Path) -> np.ndarray:
        """Extract MFCC features from audio file."""
        try:
            y, sr = librosa.load(audio_path, duration=3, offset=0.5)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            return np.expand_dims(mfcc_scaled, axis=(0, -1))
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None

    def predict_emotion(self, audio_path: Path) -> Tuple[Optional[str], float]:
        """Predict emotion from audio file."""
        if self.model is None:
            return None, 0.0
            
        features = self.extract_features(audio_path)
        if features is None:
            return None, 0.0
            
        try:
            predictions = self.model.predict(features)[0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            return self.emotion_labels[emotion_idx], confidence
        except Exception as e:
            logger.error(f"Error predicting speech emotion: {e}")
            return None, 0.0

class SentimentAnalyzer:
    """Handles text sentiment analysis using NLTK."""
    
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except:
            logger.error("Failed to initialize SentimentIntensityAnalyzer")
            self.analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not self.analyzer or not text:
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "sentiment": "Neutral"
            }
        
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Determine sentiment category
            if scores['compound'] >= 0.05:
                sentiment = "Positive"
            elif scores['compound'] <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            return {
                "compound": scores['compound'],
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "sentiment": sentiment
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "sentiment": "Neutral"
            }

class EmotionFusion:
    """Combines facial and speech emotion predictions."""
    
    @staticmethod
    def fuse_emotions(face_emotion: str, face_confidence: float,
                     speech_emotion: str, speech_confidence: float,
                     text_sentiment: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fuse facial, speech emotions and text sentiment with confidence weighting."""
        if face_emotion == "Unknown" and speech_emotion is None and (text_sentiment is None or text_sentiment.get("sentiment") == "Neutral"):
            return {
                "primary_emotion": "Unknown",
                "confidence": 0.0,
                "face_emotion": face_emotion,
                "speech_emotion": speech_emotion or "Unknown",
                "text_sentiment": text_sentiment.get("sentiment") if text_sentiment else "Unknown"
            }
        
        # Initialize weights
        face_weight = 0.0
        speech_weight = 0.0
        text_weight = 0.0
        
        # Calculate total confidence
        total_confidence = face_confidence + speech_confidence
        if text_sentiment and text_sentiment.get("compound") != 0:
            # Add text sentiment confidence (absolute value of compound score)
            text_confidence = abs(text_sentiment.get("compound", 0))
            total_confidence += text_confidence
            text_weight = text_confidence / total_confidence if total_confidence > 0 else 0.33
        
        # Normalize weights
        if total_confidence > 0:
            face_weight = face_confidence / total_confidence
            speech_weight = speech_confidence / total_confidence
        else:
            # Equal weights if no confidence data
            weights_count = sum(1 for w in [face_emotion != "Unknown", speech_emotion is not None, 
                                          text_sentiment is not None and text_sentiment.get("sentiment") != "Neutral"] 
                              if w)
            weight = 1.0 / max(weights_count, 1)
            face_weight = weight if face_emotion != "Unknown" else 0
            speech_weight = weight if speech_emotion is not None else 0
            text_weight = weight if text_sentiment is not None and text_sentiment.get("sentiment") != "Neutral" else 0
        
        # Map text sentiment to emotion
        text_emotion = None
        if text_sentiment:
            sentiment = text_sentiment.get("sentiment")
            if sentiment == "Positive":
                text_emotion = "Happy"
            elif sentiment == "Negative":
                # Use face or speech emotion if available for negative emotions
                if face_emotion in ["Sad", "Angry", "Fear", "Disgust"]:
                    text_emotion = face_emotion
                elif speech_emotion in ["Sad", "Angry"]:
                    text_emotion = speech_emotion
                else:
                    text_emotion = "Sad"  # Default negative emotion
            elif sentiment == "Neutral":
                text_emotion = "Neutral"
        
        # Determine primary emotion based on weights
        emotions = []
        if face_weight > 0:
            emotions.append((face_emotion, face_weight))
        if speech_weight > 0:
            emotions.append((speech_emotion, speech_weight))
        if text_weight > 0 and text_emotion:
            emotions.append((text_emotion, text_weight))
        
        # Sort by weight (highest first)
        emotions.sort(key=lambda x: x[1], reverse=True)
        
        # Use the emotion with highest weight
        if emotions:
            primary_emotion, confidence = emotions[0]
        else:
            primary_emotion, confidence = "Unknown", 0.0
            
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "face_emotion": face_emotion,
            "speech_emotion": speech_emotion,
            "text_sentiment": text_sentiment.get("sentiment") if text_sentiment else None
        }

class PsychiatristBot:
    """Handles interaction with Gemini AI for psychiatric responses."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.conversation_history = []
        self.init_psychiatric_context()
    
    def init_psychiatric_context(self):
        context = """You are a concise, empathetic psychiatric AI assistant. 
        Provide brief, supportive responses that offer practical emotional guidance. 
        Maintain a warm, professional tone. Focus on understanding, validating feelings, 
        and providing constructive perspectives."""
        self.conversation_history = [{"role": "system", "content": context}]
    
    def analyze_and_respond(self, 
                          transcript: str,
                          emotion_data: Dict[str, Any],
                          session_history: List[Dict] = None) -> Dict[str, Any]:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Enhanced prompt with mood-based recommendation request
            analysis_prompt = f"""
            Based on the following:
            - Primary Emotion: {emotion_data['primary_emotion']} (Confidence: {emotion_data['confidence']:.2f})
            - Patient's Statement: {transcript}
            
            Provide a brief 4-5 line psychiatric response that includes:
            1. Validation of their feelings
            2. A supportive perspective
            3. One specific recommendation based on their current emotional state
            
            Be concise but supportive.
            """
            
            if session_history:
                self.conversation_history.extend(session_history)
            
            response = self.model.generate_content(analysis_prompt)
            psychiatric_response = response.text
            
            self.conversation_history.extend([
                {
                    "role": "user",
                    "content": transcript,
                    "emotion_data": emotion_data,
                    "timestamp": current_time
                },
                {
                    "role": "assistant",
                    "content": psychiatric_response,
                    "timestamp": current_time
                }
            ])
            
            return {
                "response": psychiatric_response,
                "emotion_data": emotion_data,
                "success": True,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Error generating psychiatric response: {e}")
            return {
                "error": "Failed to generate psychiatric response",
                "success": False
            }
    
    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    def get_progress_stats(self) -> Dict[str, Any]:
        try:
            emotions_over_time = []
            emotion_counts = {}
            timestamps = []

            for entry in self.conversation_history:
                if entry["role"] == "user" and "emotion_data" in entry:
                    emotion = entry["emotion_data"]["primary_emotion"]
                    confidence = entry["emotion_data"]["confidence"]
                    timestamp = entry["timestamp"]
                    
                    emotions_over_time.append({
                        "emotion": emotion,
                        "confidence": confidence,
                        "timestamp": timestamp
                    })
                    timestamps.append(timestamp)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            total_interactions = len(emotions_over_time)
            positive_emotions = sum(1 for e in emotions_over_time if e["emotion"] in ["Happy", "Neutral"])
            negative_emotions = total_interactions - positive_emotions
            
            return {
                "emotions_over_time": emotions_over_time,
                "emotion_counts": emotion_counts,
                "total_interactions": total_interactions,
                "positive_percentage": (positive_emotions / total_interactions * 100) if total_interactions > 0 else 0,
                "negative_percentage": (negative_emotions / total_interactions * 100) if total_interactions > 0 else 0,
                "timestamps": timestamps,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error calculating progress stats: {e}")
            return {"error": str(e), "status": "error"}
    
    def get_mood_based_recommendations(self, emotion: str) -> List[str]:
        """Generate personalized recommendations based on emotional state."""
        recommendations = []
        
        if emotion == "Sad":
            recommendations = [
                "Try a brief mindfulness meditation to acknowledge your feelings without judgment.",
                "Consider reaching out to a trusted friend or family member for support.",
                "Engage in a small creative activity that you enjoy, like drawing or writing.",
                "Take a gentle walk outside to change your environment and get some fresh air.",
                "Listen to uplifting music that resonates with you."
            ]
        elif emotion == "Angry":
            recommendations = [
                "Practice deep breathing exercises: inhale for 4 counts, hold for 4, exhale for 6.",
                "Write down what's bothering you to externalize your thoughts.",
                "Engage in physical activity to release tension, like a brisk walk or stretching.",
                "Try the 5-4-3-2-1 grounding technique: identify 5 things you see, 4 you can touch, etc.",
                "Give yourself permission to take a timeout before responding to the situation."
            ]
        elif emotion == "Fear" or emotion == "Anxiety":
            recommendations = [
                "Focus on what you can control in this moment, rather than uncertainties.",
                "Practice progressive muscle relaxation by tensing and releasing each muscle group.",
                "Create a safe space where you can feel secure and comfortable.",
                "Break down overwhelming tasks into smaller, manageable steps.",
                "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8."
            ]
        elif emotion == "Happy":
            recommendations = [
                "Savor this positive feeling by journaling what contributed to your happiness.",
                "Share your positive energy with someone else through a kind gesture.",
                "Engage in activities that maintain this positive state.",
                "Practice gratitude by noting three things you appreciate right now.",
                "Set an intention to carry this feeling forward into challenging situations."
            ]
        else:  # Neutral or other emotions
            recommendations = [
                "Take a moment to check in with yourself and identify any subtle feelings.",
                "Consider what activities might enhance your current state.",
                "Practice mindfulness to stay present and aware of your emotional state.",
                "Set an intention for how you'd like to feel for the rest of the day.",
                "Engage in a self-care activity that nurtures your overall wellbeing."
            ]
        
        return recommendations

class SpeechProcessor:
    """Enhanced speech recognition with multiple engines and fallback options."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engines = ['google', 'sphinx']  # Added offline fallback
        self.current_engine = 'google'
    
    def optimize_audio(self, audio_data):
        """Optimize audio for better recognition."""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply noise reduction
            audio_data = librosa.effects.preemphasis(audio_data)
            
            return audio_data
        except Exception as e:
            logger.error(f"Audio optimization error: {e}")
            return audio_data

    def transcribe_audio(self, audio_path: Path) -> Tuple[str, bool]:
        """Transcribe audio with fallback options."""
        for engine in self.engines:
            try:
                with sr.AudioFile(str(audio_path)) as source:
                    audio = self.recognizer.record(source)
                    
                    if engine == 'google':
                        try:
                            text = self.recognizer.recognize_google(audio)
                            return text, True
                        except requests.exceptions.RequestException:
                            logger.warning("Google Speech API connection failed, trying next engine")
                            continue
                    elif engine == 'sphinx':
                        try:
                            text = self.recognizer.recognize_sphinx(audio)
                            return text, True
                        except sr.UnknownValueError:
                            continue
                        
            except Exception as e:
                logger.error(f"Error with {engine} recognition: {e}")
                continue
                
        return "Speech recognition failed", False

class AudioProcessor:
    """Handles audio processing and speech-to-text conversion."""
    
    def __init__(self, psychiatrist_bot, tts_engine):
        self.speech_processor = SpeechProcessor()
        self.recognizer = sr.Recognizer()
        self.emotion_detector = None
        self.speech_emotion_detector = SpeechEmotionDetector()
        self.psychiatrist_bot = psychiatrist_bot
        self.tts_engine = tts_engine  # Store the tts_engine
        self.emotion_fusion = EmotionFusion()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    def set_emotion_detector(self, detector: EmotionDetector):
        self.emotion_detector = detector
    
    def process_audio(self, audio_path: Path) -> Dict[str, Any]:
        try:
            wav_path = audio_path.with_suffix('.wav')
            if not AudioUtils.convert_audio_to_wav(audio_path, wav_path):
                return {'error': 'Audio conversion failed', 'success': False}

            transcript, speech_success = self.speech_processor.transcribe_audio(wav_path)
            speech_emotion, speech_confidence = self.speech_emotion_detector.predict_emotion(wav_path)

            face_emotion = self.emotion_detector.last_emotion if self.emotion_detector else "Unknown"
            face_confidence = self.emotion_detector.emotion_confidence if self.emotion_detector else 0.0
            
            # Add sentiment analysis
            text_sentiment = self.sentiment_analyzer.analyze_sentiment(transcript)

            emotion_data = self.emotion_fusion.fuse_emotions(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence,
                text_sentiment
            )

            response_data = self.psychiatrist_bot.analyze_and_respond(
                transcript,
                emotion_data
            )

            # Generate audio response
            audio_data = None
            if response_data.get('response'):
                audio_data = self.tts_engine.generate_speech(response_data['response'])

            return {
                'transcript': transcript,
                'emotion_data': emotion_data,
                'psychiatric_response': response_data.get('response'),
                'audio_data': audio_data,
                'success': True
            }
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {'error': str(e), 'success': False}
        finally:
            if wav_path.exists():
                try:
                    wav_path.unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {e}")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text input without audio."""
        try:
            face_emotion = self.emotion_detector.last_emotion if self.emotion_detector else "Unknown"
            face_confidence = self.emotion_detector.emotion_confidence if self.emotion_detector else 0.0
            
            # Add sentiment analysis
            text_sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            
            emotion_data = self.emotion_fusion.fuse_emotions(
                face_emotion, face_confidence,
                None, 0.0,  # No speech emotion for text input
                text_sentiment
            )

            response_data = self.psychiatrist_bot.analyze_and_respond(
                text,
                emotion_data
            )

            # Generate audio response
            audio_data = None
            if response_data.get('response'):
                audio_data = self.tts_engine.generate_speech(response_data['response'])

            return {
                'transcript': text,
                'emotion_data': emotion_data,
                'psychiatric_response': response_data.get('response'),
                'audio_data': audio_data,
                'success': True
            }
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {'error': str(e), 'success': False}

    def process_transcript_and_emotions(self, transcript: str, wav_path: Path) -> Dict[str, Any]:
        """Process transcript and emotions from audio."""
        try:
            # Get speech emotion
            speech_emotion, speech_confidence = self.speech_emotion_detector.predict_emotion(wav_path)

            # Get current facial emotion
            face_emotion = self.emotion_detector.last_emotion if self.emotion_detector else "Unknown"
            face_confidence = self.emotion_detector.emotion_confidence if self.emotion_detector else 0.0
            
            # Add sentiment analysis
            text_sentiment = self.sentiment_analyzer.analyze_sentiment(transcript)

            # Fuse emotions
            emotion_data = self.emotion_fusion.fuse_emotions(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence,
                text_sentiment
            )

            # Get psychiatric response
            response_data = self.psychiatrist_bot.analyze_and_respond(
                transcript,
                emotion_data
            )

            return {
                'transcript': transcript,
                'emotion_data': emotion_data,
                'psychiatric_response': response_data.get('response'),
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing transcript and emotions: {e}")
            return {
                'error': str(e),
                'success': False
            }

class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
        """Convert audio file to WAV format with robust handling."""
        try:
            # Use pydub for more robust audio conversion
            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_channels(1).set_frame_rate(16000)  # Standardize to mono, 16kHz
            audio.export(str(output_path), format="wav", codec="pcm_s16le")
            return True
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return False

    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        return librosa.util.normalize(audio_data)

class TextToSpeech:
    """Handles text-to-speech conversion for therapeutic responses."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    @retrying.retry(
        stop_max_attempt_number=3,
        wait_fixed=2000,
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def _generate_speech_internal(self, text: str) -> str:
        """Internal method for generating speech with retry logic."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir=self.temp_dir)
        
        try:
            # Generate speech with improved settings
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file.name)
            
            # Read audio file and convert to base64
            with open(temp_file.name, 'rb') as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                
            return audio_data
        finally:
            # Clean up temp file
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def generate_speech(self, text: str) -> str:
        """
        Convert text to speech and return as base64 audio data with fallback.
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Base64 encoded audio data or None if failed
        """
        try:
            return self._generate_speech_internal(text)
        except Exception as e:
            logger.error(f"Error generating speech after retries: {e}")
            # Fallback: Return a message indicating failure
            return None
    
    def cleanup(self):
        """Clean up temporary directory."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up TTS temp directory: {e}")

class Config:
    """Application configuration."""
    # Base paths    
    BASE_DIR = Path(os.getcwd())
    UPLOAD_FOLDER = BASE_DIR / 'Uploads'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # API Configuration
    GEMINI_API_KEY = "AIzaSyAudPFA827myAx1Zb_59IWoYsa8jZbE80k"  # Replace with your actual API key
    
    # Model paths
    FACE_EMOTION_MODEL_PATH = BASE_DIR / 'face_emotion_model.h5'
    SPEECH_EMOTION_MODEL_PATH = BASE_DIR / 'speech_emotion_model.h5'
    
    # Emotion detection settings
    EMOTION_CONFIDENCE_THRESHOLD = 0.6
    EMOTION_HISTORY_SIZE = 5
    
    # Audio processing settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 5  # seconds
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Check if API key is set
            if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
                logger.error("Please set your Gemini API key in the Config class")
                return False
            
            # Create required directories
            cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
            (cls.BASE_DIR / 'models').mkdir(parents=True, exist_ok=True)
            
            # Validate model files
            if not cls.FACE_EMOTION_MODEL_PATH.exists():
                logger.warning(f"Face emotion model not found at {cls.FACE_EMOTION_MODEL_PATH}")
                # Don't fail if model is missing, just log warning
            
            if not cls.SPEECH_EMOTION_MODEL_PATH.exists():
                logger.warning(f"Speech emotion model not found at {cls.SPEECH_EMOTION_MODEL_PATH}")
                # Don't fail if model is missing, just log warning
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    @classmethod
    def init_app(cls, app):
        """Initialize Flask application with config values."""
        app.config['UPLOAD_FOLDER'] = str(cls.UPLOAD_FOLDER)
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Database setup
def init_db():
    """Initialize SQLite database for user management and conversation history."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        emotion_data TEXT,
        audio_data TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    ''')
    
    # User settings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id TEXT PRIMARY KEY,
        enable_audio BOOLEAN DEFAULT 1,
        enable_notifications BOOLEAN DEFAULT 0,
        dark_mode BOOLEAN DEFAULT 0,
        text_size TEXT DEFAULT 'medium',
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# User management functions
def create_user(name, email, password):
    """Create a new user in the database."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    user_id = str(uuid.uuid4())
    hashed_password = generate_password_hash(password)
    
    try:
        cursor.execute(
            "INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)",
            (user_id, name, email, hashed_password)
        )
        
        # Create default settings for the user
        cursor.execute(
            "INSERT INTO user_settings (user_id) VALUES (?)",
            (user_id,)
        )
        
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        conn.rollback()
        return None
    finally:
        conn.close()

def authenticate_user(email, password):
    """Authenticate a user with email and password."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    
    if result and check_password_hash(result[1], password):
        return result[0]  # Return user_id
    return None

def get_user_by_id(user_id):
    """Get user details by ID."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "id": result[0],
            "name": result[1],
            "email": result[2]
        }
    return None

def get_user_settings(user_id):
    """Get user settings by user ID."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "enable_audio": bool(result[1]),
            "enable_notifications": bool(result[2]),
            "dark_mode": bool(result[3]),
            "text_size": result[4]
        }
    return None

def update_user_settings(user_id, settings):
    """Update user settings."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            UPDATE user_settings 
            SET enable_audio = ?, enable_notifications = ?, dark_mode = ?, text_size = ?
            WHERE user_id = ?
            """,
            (
                settings.get("enable_audio", True),
                settings.get("enable_notifications", False),
                settings.get("dark_mode", False),
                settings.get("text_size", "medium"),
                user_id
            )
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Conversation management functions
def create_conversation(user_id):
    """Create a new conversation for a user."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    conversation_id = str(uuid.uuid4())
    
    try:
        cursor.execute(
            "INSERT INTO conversations (id, user_id) VALUES (?, ?)",
            (conversation_id, user_id)
        )
        
        conn.commit()
        return conversation_id
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_user_conversations(user_id):
    """Get all conversations for a user."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    
    conversations = []
    for row in cursor.fetchall():
        conversation_id = row[0]
        
        # Get the first message of each conversation
        cursor.execute(
            "SELECT content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT 1",
            (conversation_id,)
        )
        
        first_message = cursor.fetchone()
        preview = first_message[0] if first_message else "Empty conversation"
        
        conversations.append({
            "id": conversation_id,
            "created_at": row[1],
            "preview": preview[:50] + "..." if len(preview) > 50 else preview
        })
    
    conn.close()
    return conversations

def save_message(conversation_id, role, content, emotion_data=None, audio_data=None):
    """Save a message to the database."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    message_id = str(uuid.uuid4())
    emotion_json = json.dumps(emotion_data) if emotion_data else None
    
    try:
        cursor.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, emotion_data, audio_data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, emotion_json, audio_data)
        )
        
        conn.commit()
        return message_id
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_conversation_messages(conversation_id):
    """Get all messages for a conversation."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, role, content, emotion_data, audio_data, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        """,
        (conversation_id,)
    )
    
    messages = []
    for row in cursor.fetchall():
        emotion_data = json.loads(row[3]) if row[3] else None
        
        messages.append({
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "emotion_data": emotion_data,
            "audio_data": row[4],
            "timestamp": row[5]
        })
    
    conn.close()
    return messages

def get_user_emotion_stats(user_id):
    """Get emotion statistics for a user."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT m.emotion_data, m.timestamp
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.emotion_data IS NOT NULL
        ORDER BY m.timestamp ASC
        """,
        (user_id,)
    )
    
    emotions_over_time = []
    emotion_counts = {}
    timestamps = []
    
    for row in cursor.fetchall():
        if row[0]:
            emotion_data = json.loads(row[0])
            if "primary_emotion" in emotion_data and "confidence" in emotion_data:
                emotion = emotion_data["primary_emotion"]
                confidence = emotion_data["confidence"]
                timestamp = row[1]
                
                emotions_over_time.append({
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": timestamp
                })
                
                timestamps.append(timestamp)
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    conn.close()
    
    total_interactions = len(emotions_over_time)
    positive_emotions = sum(1 for e in emotions_over_time if e["emotion"] in ["Happy", "Neutral"])
    negative_emotions = total_interactions - positive_emotions
    
    return {
        "emotions_over_time": emotions_over_time,
        "emotion_counts": emotion_counts,
        "total_interactions": total_interactions,
        "positive_percentage": (positive_emotions / total_interactions * 100) if total_interactions > 0 else 0,
        "negative_percentage": (negative_emotions / total_interactions * 100) if total_interactions > 0 else 0,
        "timestamps": timestamps,
        "status": "success"
    }

def export_conversation_history(user_id, format="json"):
    """Export conversation history for a user."""
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT c.id, c.created_at
        FROM conversations c
        WHERE c.user_id = ?
        ORDER BY c.created_at DESC
        """,
        (user_id,)
    )
    
    conversations = []
    for conv_row in cursor.fetchall():
        conversation_id = conv_row[0]
        
        cursor.execute(
            """
            SELECT role, content, emotion_data, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conversation_id,)
        )
        
        messages = []
        for msg_row in cursor.fetchall():
            emotion_data = json.loads(msg_row[2]) if msg_row[2] else None
            
            messages.append({
                "role": msg_row[0],
                "content": msg_row[1],
                "emotion_data": emotion_data,
                "timestamp": msg_row[3]
            })
        
        conversations.append({
            "id": conversation_id,
            "created_at": conv_row[1],
            "messages": messages
        })
    
    conn.close()
    
    if format == "json":
        return json.dumps(conversations, indent=2)
    else:
        # For future support of other formats like CSV or PDF
        return json.dumps(conversations, indent=2)

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(Path(os.getcwd()) / 'Uploads')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 7  # 7 days

# Initialize Flask-Session
Session(app)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('sessions', exist_ok=True)

# Initialize database
init_db()

# Function to initialize application components
def initialize_components():
    try:
        if not Config.validate_config():
            raise RuntimeError("Invalid configuration. Please check the logs for details.")
        
        psychiatrist_bot = PsychiatristBot(api_key=Config.GEMINI_API_KEY)
        emotion_detector = EmotionDetector()
        tts_engine = TextToSpeech()
        audio_processor = AudioProcessor(psychiatrist_bot, tts_engine)
        audio_processor.set_emotion_detector(emotion_detector)
        
        logger.info("All components initialized successfully")
        return psychiatrist_bot, emotion_detector, audio_processor, tts_engine
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

# Initialize components
try:
    psychiatrist_bot, emotion_detector, audio_processor, tts_engine = initialize_components()
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    psychiatrist_bot = None
    emotion_detector = None
    audio_processor = None
    tts_engine = None

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not all(k in data for k in ['name', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    user_id = create_user(data['name'], data['email'], data['password'])
    
    if not user_id:
        return jsonify({'error': 'Email already registered'}), 409
    
    session['user_id'] = user_id
    
    return jsonify({
        'success': True,
        'user': {
            'id': user_id,
            'name': data['name'],
            'email': data['email']
        }
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not all(k in data for k in ['email', 'password']):
        return jsonify({'error': 'Missing email or password'}), 400
    
    user_id = authenticate_user(data['email'], data['password'])
    
    if not user_id:
        return jsonify({'error': 'Invalid email or password'}), 401
    
    user = get_user_by_id(user_id)
    session['user_id'] = user_id
    
    return jsonify({
        'success': True,
        'user': user
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/auth/user', methods=['GET'])
def get_current_user():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'authenticated': False}), 401
    
    user = get_user_by_id(user_id)
    
    if not user:
        session.pop('user_id', None)
        return jsonify({'authenticated': False}), 401
    
    return jsonify({
        'authenticated': True,
        'user': user
    })

# User settings routes
@app.route('/api/user/settings', methods=['GET'])
def get_settings():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    settings = get_user_settings(user_id)
    
    if not settings:
        return jsonify({'error': 'Settings not found'}), 404
    
    return jsonify({
        'success': True,
        'settings': settings
    })

@app.route('/api/user/settings', methods=['PUT'])
def update_settings():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No settings provided'}), 400
    
    success = update_user_settings(user_id, data)
    
    if not success:
        return jsonify({'error': 'Failed to update settings'}), 500
    
    return jsonify({
        'success': True,
        'settings': get_user_settings(user_id)
    })

# Conversation routes
@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conversations = get_user_conversations(user_id)
    
    return jsonify({
        'success': True,
        'conversations': conversations
    })

@app.route('/api/conversations', methods=['POST'])
def create_new_conversation():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conversation_id = create_conversation(user_id)
    
    if not conversation_id:
        return jsonify({'error': 'Failed to create conversation'}), 500
    
    return jsonify({
        'success': True,
        'conversation_id': conversation_id
    })

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    messages = get_conversation_messages(conversation_id)
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/export_conversation', methods=['POST'])
def export_conversation():
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    format = data.get('format', 'json') if data else 'json'
    
    history = export_conversation_history(user_id, format)
    
    if format == 'json':
        return Response(
            history,
            mimetype='application/json',
            headers={
                'Content-Disposition': 'attachment;filename=conversation_history.json'
            }
        )
    else:
        return jsonify({'error': 'Unsupported format'}), 400

@app.route('/export_report', methods=['POST'])
def export_report():
    user_id = session.get('user_id')
    
    # Allow non-authenticated users to export reports too
    if user_id:
        stats = get_user_emotion_stats(user_id)
    else:
        stats = psychiatrist_bot.get_progress_stats()
    
    # Create a simple text representation of the report
    report_text = f"""
    EMOFUSION Mood Report
    =====================
    
    Generated on: {datetime.now().strftime('%Y-%m-%d')}
    
    Emotional Trends Summary:
    ------------------------- 
    Total Interactions: {stats.get('total_interactions', 0)}
    Positive Emotions: {stats.get('positive_percentage', 0):.1f}%
    Negative Emotions: {stats.get('negative_percentage', 0):.1f}%
    
    Emotion Breakdown:
    -----------------
    {chr(10).join([f"{emotion}: {count}" for emotion, count in stats.get('emotion_counts', {}).items()])}
    """
    
    # Convert text to a Blob that simulates a PDF
    response = Response(
        report_text,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': 'attachment;filename=Mood_Report.pdf'
        }
    )
    
    return response

# Main routes
@app.route('/')
def index():
    """Render the home page."""
    try:
        with open('templates/index.html', 'r', encoding='utf-8', errors='replace') as file:
            template_content = file.read()
        return render_template_string(template_content)
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return "Error loading page", 500

@app.route('/app')
def app_page():
    """Render the application page after login."""
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/')
    
    try:
        with open('templates/index.html', 'r', encoding='utf-8', errors='replace') as file:
            template_content = file.read()
        return render_template_string(template_content)
    except FileNotFoundError:
        # Fallback to index.html if app.html doesn't exist
        return redirect('/')
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return "Error loading page", 500
    
@app.route('/api/auth/check-email', methods=['POST'])
def check_email():
    """Check if an email is already registered."""
    data = request.get_json()
    
    if not data or 'email' not in data:
        return jsonify({'error': 'Email is required'}), 400
    
    conn = sqlite3.connect('emofusion.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE email = ?", (data['email'],))
    result = cursor.fetchone()
    conn.close()
    
    return jsonify({'exists': result is not None})

@app.route('/video_feed')
def video_feed():
    if not emotion_detector or not emotion_detector.running:
        return Response("Camera not available", status=503)
    
    return Response(
        emotion_detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        
        user_id = session.get('user_id')
        conversation_id = request.form.get('conversation_id')
        
        # Create a new conversation if user is logged in but no conversation_id provided
        if user_id and not conversation_id:
            conversation_id = create_conversation(user_id)

        try:
            file.save(str(filepath))
            results = audio_processor.process_audio(filepath)

            if results.get('success', False):
                # Save messages to database if user is logged in
                if user_id and conversation_id:
                    save_message(
                        conversation_id, 
                        "user", 
                        results.get('transcript', ''), 
                        results.get('emotion_data')
                    )
                    
                    if results.get('psychiatric_response'):
                        audio_data = results.get('audio_data')
                        save_message(
                            conversation_id,
                            "assistant",
                            results.get('psychiatric_response', ''),
                            None,
                            audio_data
                        )
                
                return jsonify({
                    'transcript': results.get('transcript', ''),
                    'emotion_data': results.get('emotion_data', {}),
                    'psychiatric_response': results.get('psychiatric_response', ''),
                    'audio_data': results.get('audio_data'),
                    'conversation_id': conversation_id,
                    'success': True
                })
            return jsonify({'error': results.get('error', 'Unknown error')}), 500
        finally:
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    logger.error(f"Error removing upload file: {e}")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/text_input', methods=['POST'])
def text_input():
    """Handle text input from the frontend."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text input'}), 400
        
        user_id = session.get('user_id')
        conversation_id = data.get('conversation_id')
        
        # Create a new conversation if user is logged in but no conversation_id provided
        if user_id and not conversation_id:
            conversation_id = create_conversation(user_id)

        results = audio_processor.process_text(text)
        if results.get('success', False):
            # Save messages to database if user is logged in
            if user_id and conversation_id:
                save_message(
                    conversation_id, 
                    "user", 
                    results.get('transcript', ''), 
                    results.get('emotion_data')
                )
                
                if results.get('psychiatric_response'):
                    audio_data = results.get('audio_data')
                    save_message(
                        conversation_id,
                        "assistant",
                        results.get('psychiatric_response', ''),
                        None,
                        audio_data
                    )
            
            return jsonify({
                'transcript': results.get('transcript', ''),
                'emotion_data': results.get('emotion_data', {}),
                'psychiatric_response': results.get('psychiatric_response', ''),
                'audio_data': results.get('audio_data'),
                'conversation_id': conversation_id,
                'success': True
            })
        return jsonify({'error': results.get('error', 'Unknown error')}), 500
    except Exception as e:
        logger.error(f"Text input error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress_stats')
def get_progress_stats():
    try:
        user_id = session.get('user_id')
        
        if user_id:
            # Get stats from database for logged in users
            stats = get_user_emotion_stats(user_id)
        else:
            # Get stats from memory for non-logged in users
            stats = psychiatrist_bot.get_progress_stats()
            
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching progress stats: {e}")
        return jsonify({"error": "Failed to fetch progress stats", "status": "error"}), 500

@app.route('/psychiatric_history')
def get_psychiatric_history():
    try:
        user_id = session.get('user_id')
        
        if user_id:
            # Get all conversations for the user
            conversations = get_user_conversations(user_id)
            
            # Get messages from the most recent conversation
            if conversations:
                most_recent_conversation_id = conversations[0]['id']
                history = get_conversation_messages(most_recent_conversation_id)
            else:
                history = []
        else:
            # Get history from memory for non-logged in users
            history = psychiatrist_bot.get_conversation_history()
            
        return jsonify({'history': history, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error fetching psychiatric history: {e}")
        return jsonify({'error': 'Failed to fetch history', 'status': 'error'}), 500

@app.route('/emotion_status')
def get_emotion_status():
    try:
        return jsonify({
            'face_emotion': emotion_detector.last_emotion if emotion_detector else "Unknown",
            'face_confidence': emotion_detector.emotion_confidence if emotion_detector else 0.0,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting emotion status: {e}")
        return jsonify({'error': 'Failed to get emotion status', 'status': 'error'}), 500

@app.route('/stop_camera')
def stop_camera():
    try:
        if emotion_detector:
            emotion_detector.cleanup()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({'error': 'Failed to stop camera', 'status': 'error'}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get mood-based recommendations for a specific emotion."""
    try:
        data = request.get_json()
        if not data or 'emotion' not in data:
            return jsonify({'error': 'No emotion provided'}), 400
            
        emotion = data['emotion']
        recommendations = psychiatrist_bot.get_mood_based_recommendations(emotion)
        
        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations,
            'success': True
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    """Generate speech from text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        audio_data = tts_engine.generate_speech(text)
        
        return jsonify({
            'audio_data': audio_data,
            'success': True
        })
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large', 'max_size': app.config['MAX_CONTENT_LENGTH']}), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Application startup error: {e}")
    finally:
        if 'emotion_detector' in locals() and emotion_detector:
            emotion_detector.cleanup()
        if 'tts_engine' in locals() and tts_engine:
            tts_engine.cleanup()