import cv2
import numpy as np
from deepface import DeepFace
from database import DatabaseManager
import time
import threading
from datetime import datetime, time as dt_time
import json

class FaceRecognitionEngine:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.known_embeddings = {}
        self.known_names = {}
        self.load_known_faces()
        self.is_running = False
        self.current_checkin_type = 1
        
    def load_known_faces(self):
        """Load known face embeddings from database"""
        users = self.db_manager.get_all_users()
        for user in users:
            if user.face_embedding:
                embedding = np.array(json.loads(user.face_embedding))
                self.known_embeddings[user.user_id] = embedding
                self.known_names[user.user_id] = user.name
    
    def add_new_face(self, name, image):
        """Add a new face to the system"""
        try:
            # Generate embedding using DeepFace
            embedding_obj = DeepFace.represent(
                image, 
                model_name='Facenet', 
                enforce_detection=False
            )
            
            if embedding_obj:
                embedding = np.array(embedding_obj[0]['embedding'])
                user_id = self.db_manager.add_user(name, embedding)
                
                # Update local cache
                self.known_embeddings[user_id] = embedding
                self.known_names[user_id] = name
                
                return user_id
        except Exception as e:
            print(f"Error adding new face: {e}")
            return None
    
    def identify_face(self, image):
        """Identify face in the given image"""
        try:
            # Get embedding from current face
            embedding_obj = DeepFace.represent(
                image, 
                model_name='Facenet', 
                enforce_detection=False
            )
            
            if not embedding_obj:
                return None, None
            
            current_embedding = np.array(embedding_obj[0]['embedding'])
            
            # Compare with known embeddings
            min_distance = float('inf')
            best_match_id = None
            
            for user_id, known_embedding in self.known_embeddings.items():
                distance = np.linalg.norm(current_embedding - known_embedding)
                if distance < min_distance and distance < 0.6:  # Threshold for FaceNet
                    min_distance = distance
                    best_match_id = user_id
            
            return best_match_id, min_distance if best_match_id else None
            
        except Exception as e:
            print(f"Error in face identification: {e}")
            return None, None
    
    def determine_checkin_type(self):
        """Determine current check-in type based on time"""
        current_time = datetime.now().time()
        
        # Morning Start: 8:00 AM - 10:00 AM
        if dt_time(8, 0) <= current_time <= dt_time(10, 0):
            return 1
        # Lunch Break Out: 12:00 PM - 1:00 PM
        elif dt_time(12, 0) <= current_time <= dt_time(13, 0):
            return 2
        # Lunch Break In: 1:00 PM - 2:00 PM
        elif dt_time(13, 0) <= current_time <= dt_time(14, 0):
            return 3
        # Evening End: 5:00 PM - 7:00 PM
        elif dt_time(16, 0) <= current_time <= dt_time(19, 0):
            return 4
        else:
            return None
    
    def process_frame(self, frame):
        """Process a single video frame for face recognition"""
        self.current_checkin_type = self.determine_checkin_type()
        
        if self.current_checkin_type is None:
            return frame, "Outside check-in hours", None
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        user_id, distance = self.identify_face(small_frame)
        
        if user_id:
            name = self.known_names[user_id]
            
            # Record attendance
            self.db_manager.record_attendance(user_id, self.current_checkin_type)
            
            # Draw bounding box and info
            cv2.putText(frame, f"Identified: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Check-in: {self.current_checkin_type}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return frame, f"Welcome {name}! Check-in {self.current_checkin_type} recorded.", user_id
        else:
            cv2.putText(frame, "Unknown Face", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, "Unknown face - not registered", None
    
    def start_live_recognition(self):
        """Start live face recognition from webcam"""
        self.is_running = True
        cap = cv2.VideoCapture(0)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, status, user_id = self.process_frame(frame)
            
            cv2.imshow('Face Recognition Attendance System', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def stop_live_recognition(self):
        """Stop live recognition"""
        self.is_running = False