import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional
import pickle
import base64
from ..utils.database import db

class FaceRecognitionSystem:
    def __init__(self, tolerance: float = 0.6):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.face_locations = []
        self.face_encodings = []
        self.process_this_frame = True
        self.tolerance = tolerance
        self.load_known_faces()

    def load_known_faces(self):
        """Load known face encodings from the database."""
        session = db.get_session()
        try:
            users = session.query(db.User).filter_by(is_active=True).all()
            for user in users:
                # Convert string representation back to numpy array
                face_encoding = pickle.loads(base64.b64decode(user.face_encoding))
                self.known_face_encodings.append(face_encoding)
                self.known_face_ids.append(user.id)
        finally:
            session.close()

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the given frame."""
        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Scale back up face locations
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                         for (top, right, bottom, left) in face_locations]
        
        return face_locations

    def get_face_encodings(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Get face encodings for detected faces."""
        rgb_frame = frame[:, :, ::-1]
        return face_recognition.face_encodings(rgb_frame, face_locations)

    def identify_face(self, face_encoding: np.ndarray) -> Tuple[Optional[int], float]:
        """Identify a face from known faces."""
        if not self.known_face_encodings:
            return None, 0.0

        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding, 
            tolerance=self.tolerance
        )
        
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            return self.known_face_ids[best_match_index], 1 - face_distances[best_match_index]
        
        return None, 0.0

    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Process a frame and return identified faces with their locations and confidence."""
        if self.process_this_frame:
            # Detect faces
            self.face_locations = self.detect_faces(frame)
            
            # Get face encodings
            self.face_encodings = self.get_face_encodings(frame, self.face_locations)
            
            self.process_this_frame = False
        
        results = []
        for face_encoding, face_location in zip(self.face_encodings, self.face_locations):
            user_id, confidence = self.identify_face(face_encoding)
            if user_id is not None:
                results.append((user_id, face_location, confidence))
        
        self.process_this_frame = not self.process_this_frame
        return results

    def encode_face(self, image_path: str) -> Optional[np.ndarray]:
        """Encode a face from an image file."""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                return None
            
            # Return the first face encoding
            return face_encodings[0]
        except Exception as e:
            print(f"Error encoding face: {e}")
            return None

    def draw_results(self, frame: np.ndarray, results: List[Tuple[int, Tuple[int, int, int, int], float]]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        for user_id, (top, right, bottom, left), confidence in results:
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label
            label = f"User {user_id} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame 