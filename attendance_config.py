# attendance_config.py
DB_CONFIG = {
    'host': 'localhost',
    'database': 'attendance2',
    'user': 'postgres',
    'password': 'Pass@123'
}

FACE_RECOGNITION_CONFIG = {
    'tolerance': 6000,
    'model': 'opencv',
    'num_jitters': 1,
    'detection_scale': 0.5,
    'encoding_scale': 1.0,
}

ATTENDANCE_COOLDOWN = 300

CAMERA_CONFIG = {
    'default_camera': 0,
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
