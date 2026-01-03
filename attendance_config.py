# attendance_config.py
"""
Configuration file for the Face Recognition Attendance System
"""

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'attendance2',
    'user': 'postgres',
    'password': 'Pass@123'
}

# Face Recognition Settings
FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.5,  # Lower = stricter matching (0.6 is default, 0.4-0.5 recommended for accuracy)
    'model': 'hog',  # 'hog' for CPU, 'cnn' for GPU (more accurate but slower)
    'num_jitters': 2,  # Number of times to re-sample face for encoding (higher = more accurate but slower)
    'detection_scale': 0.5,  # Scale for real-time detection (0.5 = 50% of original size)
    'encoding_scale': 1.0,  # Scale for encoding during registration (1.0 = full resolution)
}

# Attendance Settings
ATTENDANCE_COOLDOWN = 300  # 5 minutes cooldown between attendance logs

# Camera Settings
CAMERA_CONFIG = {
    'default_camera': 0,  # 0 for built-in webcam
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30
}

# UI Settings
UI_CONFIG = {
    'theme': 'default',  # ttk theme
    'primary_color': '#2196F3',
    'success_color': '#4CAF50',
    'error_color': '#F44336',
    'warning_color': '#FF9800'
}
