# face_recognition_engine.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

# Import and initialize DeepFace early
print("Loading DeepFace models... (this may take 30 seconds)")
from deepface import DeepFace
from deepface.commons import functions

class FaceRecognitionEngine:
    def __init__(self, config):
        self.config = config
        self.known_encodings = []
        self.known_prns = []
        self.lock = Lock()
        
        # Pre-build model to avoid hanging later
        try:
            print(f"Building {config['model']} model...")
            DeepFace.build_model(config['model'])
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")
        
    def load_known_faces(self, encodings, prns):
        """Load known face encodings into memory"""
        with self.lock:
            self.known_encodings = encodings
            self.known_prns = prns
        print(f"✓ Loaded {len(prns)} face encodings")

    def detect_and_encode_face(self, frame, for_registration=False):
        """Detect faces and generate encodings using DeepFace"""
        scale = self.config['encoding_scale'] if for_registration else self.config['detection_scale']
        
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame
            
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use opencv backend (faster)
            face_objs = DeepFace.extract_faces(
                rgb_frame, 
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            
            face_locations = []
            face_encodings = []
            
            for face_obj in face_objs:
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Skip very small faces
                if w < 30 or h < 30:
                    continue
                
                # Convert to (top, right, bottom, left)
                top, right, bottom, left = y, x+w, y+h, x
                if scale != 1.0:
                    top = int(top/scale)
                    right = int(right/scale)
                    bottom = int(bottom/scale)
                    left = int(left/scale)
                
                face_locations.append((top, right, bottom, left))
                
                # Extract face for embedding
                face_img = rgb_frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                
                # Get embedding
                embedding_objs = DeepFace.represent(
                    face_img,
                    model_name=self.config['model'],
                    enforce_detection=False
                )
                
                if embedding_objs:
                    embedding = embedding_objs[0]['embedding']
                    face_encodings.append(np.array(embedding))
            
            return face_locations, face_encodings
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], []

    def recognize_faces(self, face_encodings):
        """Match face encodings against known faces"""
        results = []
        
        with self.lock:
            if len(self.known_encodings) == 0:
                return [(None, 0.0) for _ in face_encodings]
            
            for face_encoding in face_encodings:
                distances = [np.linalg.norm(face_encoding - known) for known in self.known_encodings]
                
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                
                confidence = max(0, (1 - best_distance/2) * 100)
                
                if best_distance <= (self.config['tolerance'] * 2):
                    prn = self.known_prns[best_match_index]
                    results.append((prn, confidence))
                else:
                    results.append((None, confidence))
        
        return results

    def enhance_image_quality(self, frame):
        """Apply image enhancement"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced
