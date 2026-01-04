# face_recognition_engine.py - Simple OpenCV Version
import cv2
import numpy as np
from threading import Lock

class FaceRecognitionEngine:
    def __init__(self, config):
        self.config = config
        self.known_encodings = []
        self.known_prns = []
        self.lock = Lock()
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Face recognition initialized")
        
    def load_known_faces(self, encodings, prns):
        with self.lock:
            self.known_encodings = encodings
            self.known_prns = prns
        print(f"✓ Loaded {len(prns)} face encodings")

    def detect_and_encode_face(self, frame, for_registration=False):
        scale = self.config['detection_scale']
        
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame
            
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        face_locations = []
        face_encodings = []
        
        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x+w, y+h, x
            
            if scale != 1.0:
                top = int(top/scale)
                right = int(right/scale)
                bottom = int(bottom/scale)
                left = int(left/scale)
            
            face_locations.append((top, right, bottom, left))
            
            # Create simple encoding from face region
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            face_encodings.append(face_resized.flatten().astype(np.float64))
        
        return face_locations, face_encodings

    def recognize_faces(self, face_encodings):
        results = []
        
        with self.lock:
            if len(self.known_encodings) == 0:
                return [(None, 0.0) for _ in face_encodings]
            
            for face_encoding in face_encodings:
                distances = [np.linalg.norm(face_encoding - known) 
                           for known in self.known_encodings]
                
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                
                # Confidence calculation
                confidence = max(0, (1 - best_dist/8000) * 100)
                
                # Threshold
                if best_dist < 6000:
                    prn = self.known_prns[best_idx]
                    results.append((prn, confidence))
                else:
                    results.append((None, confidence))
        
        return results

    def enhance_image_quality(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
