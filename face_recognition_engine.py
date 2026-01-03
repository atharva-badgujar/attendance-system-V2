# face_recognition_engine.py
"""
Enhanced face recognition with better accuracy
"""

import cv2
import face_recognition
import numpy as np
from threading import Lock

class FaceRecognitionEngine:
    def __init__(self, config):
        self.config = config
        self.known_encodings = []
        self.known_prns = []
        self.lock = Lock()
        
    def load_known_faces(self, encodings, prns):
        """Load known face encodings into memory"""
        with self.lock:
            self.known_encodings = encodings
            self.known_prns = prns
        print(f"✓ Loaded {len(prns)} face encodings")

    def detect_and_encode_face(self, frame, for_registration=False):
        """
        Detect faces and generate encodings
        Returns: (face_locations, face_encodings)
        """
        scale = self.config['encoding_scale'] if for_registration else self.config['detection_scale']
        
        # Resize frame for processing
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model=self.config['model']
        )
        
        if len(face_locations) == 0:
            return [], []
        
        # Generate encodings with jittering for better accuracy
        face_encodings = face_recognition.face_encodings(
            rgb_frame, 
            face_locations,
            num_jitters=self.config['num_jitters']
        )
        
        # Scale locations back to original size
        if scale != 1.0:
            face_locations = [(int(top/scale), int(right/scale), 
                             int(bottom/scale), int(left/scale))
                            for top, right, bottom, left in face_locations]
        
        return face_locations, face_encodings

    def recognize_faces(self, face_encodings):
        """
        Match face encodings against known faces
        Returns: list of (prn, confidence) tuples
        """
        results = []
        
        with self.lock:
            if len(self.known_encodings) == 0:
                return [(None, 0.0) for _ in face_encodings]
            
            for face_encoding in face_encodings:
                # Compare with all known faces
                face_distances = face_recognition.face_distance(
                    self.known_encodings, 
                    face_encoding
                )
                
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                # Convert distance to confidence (0-100%)
                confidence = (1 - best_distance) * 100
                
                # Check if match is good enough
                if best_distance <= self.config['tolerance']:
                    prn = self.known_prns[best_match_index]
                    results.append((prn, confidence))
                else:
                    results.append((None, confidence))
        
        return results

    def enhance_image_quality(self, frame):
        """Apply image enhancement for better recognition"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
