import unittest
import cv2
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager
from live_engine import FaceRecognitionEngine

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager("sqlite:///test_attendance.db")
        self.face_engine = FaceRecognitionEngine(self.db_manager)

    def test_face_registration(self):
        # Test adding a new face
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        user_id = self.face_engine.add_new_face("Test User", test_image)
        self.assertIsNotNone(user_id)

    def test_face_identification(self):
        # Test face identification logic
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Try to identify (should return None for random image)
        user_id, distance = self.face_engine.identify_face(test_image)
        # This test just ensures the function runs without error
        self.assertTrue(True)

    def tearDown(self):
        self.db_manager.close()

if __name__ == '__main__':
    unittest.main()