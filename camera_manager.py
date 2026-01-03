# camera_manager.py
"""
Manages multiple camera sources with auto-detection
"""

import cv2
import threading

class CameraManager:
    def __init__(self, config):
        self.config = config
        self.available_cameras = []
        self.detect_cameras()
        
    def detect_cameras(self):
        """Detect all available cameras"""
        self.available_cameras = []
        
        # Test first 5 camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    camera_info = {
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}"
                    }
                    self.available_cameras.append(camera_info)
                cap.release()
        
        # Check for RTSP/IP cameras (if URLs are provided)
        # Example: rtsp://username:password@ip:port/stream
        
        print(f"✓ Detected {len(self.available_cameras)} camera(s)")
        return self.available_cameras
    
    def open_camera(self, camera_index):
        """Open a camera with optimal settings"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return None
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        return cap
    
    def get_camera_list(self):
        """Get list of camera names for UI dropdown"""
        return [f"{cam['name']} ({cam['resolution']})" 
                for cam in self.available_cameras]
