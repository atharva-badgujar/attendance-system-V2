# face_recognition_engine.py - YOLOv8 + InsightFace Version
import cv2
import numpy as np
from threading import Lock
from ultralytics import YOLO
from insightface import app

class FaceRecognitionEngine:
    def __init__(self, config):
        self.config = config
        self.known_embeddings = []
        self.known_prns = []
        self.lock = Lock()
        self.face_analyzer = None
        self.yolo_detector = None

        self._initialize_face_analyzer()
        self._initialize_yolo_detector()
        print("✓ Face recognition initialized")

    def _initialize_face_analyzer(self):
        model_name = self.config.get('insightface_model', 'buffalo_l')
        self.face_analyzer = app.FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition']
        )
        self.face_analyzer.prepare(
            ctx_id=0,
            det_thresh=self.config.get('det_thresh', 0.5),
            det_size=tuple(self.config.get('det_size', (640, 640)))
        )
        print(f"✓ InsightFace model loaded: {model_name}")

    def _initialize_yolo_detector(self):
        if not self.config.get('use_yolo', False):
            return

        model_path = self.config.get('yolo_model', 'yolov8n.pt')
        try:
            self.yolo_detector = YOLO(model_path)
            print(f"✓ YOLO detector loaded: {model_path}")
        except Exception as exc:
            print(f"⚠ Failed to load YOLO model '{model_path}': {exc}")
            self.yolo_detector = None

    def load_known_faces(self, encodings, prns):
        with self.lock:
            normalized_embeddings = []
            for encoding in encodings:
                embedding = np.asarray(encoding, dtype=np.float32).flatten()
                normalized_embeddings.append(self._normalize_embedding(embedding))
            self.known_embeddings = normalized_embeddings
            self.known_prns = list(prns)
        print(f"✓ Loaded {len(self.known_prns)} known face embeddings")

    def _normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.astype(np.float32)
        return (embedding / norm).astype(np.float32)

    def detect_and_encode_face(self, frame, for_registration=False):
        scale = float(self.config.get('detection_scale', 1.0))
        detection_frame = frame
        if scale != 1.0:
            detection_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        face_locations = []
        face_encodings = []

        if self.yolo_detector is not None:
            face_locations, face_encodings = self._detect_with_yolo(detection_frame)
            if face_encodings:
                if scale != 1.0:
                    face_locations = [
                        (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))
                        for top, right, bottom, left in face_locations
                    ]
                return face_locations, face_encodings

        faces = self.face_analyzer.get(detection_frame)
        for face in faces:
            bbox = getattr(face, 'bbox', None)
            embedding = getattr(face, 'embedding', None)
            if bbox is None or embedding is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            top, right, bottom, left = y1, x2, y2, x1
            face_locations.append((top, right, bottom, left))
            face_encodings.append(self._normalize_embedding(np.asarray(embedding, dtype=np.float32)))

        if scale != 1.0:
            face_locations = [
                (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))
                for top, right, bottom, left in face_locations
            ]

        return face_locations, face_encodings

    def _detect_with_yolo(self, frame):
        face_locations = []
        face_encodings = []
        try:
            results = self.yolo_detector(frame)
        except Exception:
            return face_locations, face_encodings

        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            for box, cls, conf in zip(xyxy, classes, confidences):
                if int(cls) != 0 or conf < float(self.config.get('yolo_confidence', 0.3)):
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_faces = self.face_analyzer.get(crop)
                for face in crop_faces:
                    bbox = getattr(face, 'bbox', None)
                    embedding = getattr(face, 'embedding', None)
                    if bbox is None or embedding is None:
                        continue

                    fx1, fy1, fx2, fy2 = [int(v) for v in bbox]
                    top = fy1 + y1
                    right = fx2 + x1
                    bottom = fy2 + y1
                    left = fx1 + x1

                    face_locations.append((top, right, bottom, left))
                    face_encodings.append(self._normalize_embedding(np.asarray(embedding, dtype=np.float32)))

        return face_locations, face_encodings

    def recognize_faces(self, face_encodings):
        results = []
        with self.lock:
            if len(self.known_embeddings) == 0:
                return [(None, 0.0) for _ in face_encodings]

            known_stack = np.vstack(self.known_embeddings)
            threshold = float(self.config.get('recognition_tolerance', 0.8))

            for face_encoding in face_encodings:
                query = self._normalize_embedding(np.asarray(face_encoding, dtype=np.float32))
                distances = np.linalg.norm(known_stack - query, axis=1)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])

                confidence = max(0.0, min(100.0, (1.2 - best_dist) / 1.2 * 100))
                if best_dist <= threshold:
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
