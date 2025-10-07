# Configuration settings
import os

class Config:
    # Database
    DATABASE_URL = "sqlite:///attendance_system.db"
    
    # Face Recognition
    FACE_RECOGNITION_MODEL = "Facenet"
    FACE_DETECTION_THRESHOLD = 0.6
    
    # Check-in Time Windows
    CHECKIN_WINDOWS = {
        1: {"name": "Morning Start", "start": "08:00", "end": "10:00"},
        2: {"name": "Lunch Break Out", "start": "12:00", "end": "13:00"},
        3: {"name": "Lunch Break In", "start": "13:00", "end": "14:00"},
        4: {"name": "Evening End", "start": "16:40", "end": "19:00"}
    }
    
    # Streamlit Configuration
    STREAMLIT_TITLE = "Real-Time Face Recognition Attendance System"
    
    # File Paths
    MODEL_PATH = "models/face_embeddings.pkl"