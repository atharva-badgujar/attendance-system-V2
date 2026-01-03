# face_recognition_engine.py
"""
Enhanced face recognition with better accuracy using DeepFace
"""

import cv2
from deepface import DeepFace
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
        Detect faces and generate encodings using DeepFace
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
        
        try:
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(rgb_frame, detector_backend='opencv', enforce_detection=False)
            
            face_locations = []
            face_encodings = []
            
            for face in faces:
                if face['confidence'] > 0.5:
                    # Get face region
                    facial_area = face['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Convert to (top, right, bottom, left) format
                    top, right, bottom, left = y, x+w, y+h, x
                    if scale != 1.0:
                        top, right, bottom, left = int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)
                    
                    face_locations.append((top, right, bottom, left))
                    
                    # Generate embedding
                    face_img = rgb_frame[y:y+h, x:x+w]
                    if face_img.size == 0:
                        continue
                    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']
                    face_encodings.append(np.array(embedding))
            
            return face_locations, face_encodings
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return [], []

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
                # Calculate distances using Euclidean distance
                distances = [np.linalg.norm(face_encoding - known) for known in self.known_encodings]
                
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                
                # Convert distance to confidence (0-100%)
                confidence = max(0, (1 - best_distance/2) * 100)
                
                # Threshold check (DeepFace embeddings have larger distances)
                if best_distance <= (self.config['tolerance'] * 2):
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
