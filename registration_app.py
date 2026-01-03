# Main Application File (registration_app.py)
"""
Professional Registration Application with Enhanced UI
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

class RegistrationApp:
    def __init__(self, root, db_manager, face_engine, camera_manager):
        self.root = root
        self.db = db_manager
        self.face_engine = face_engine
        self.camera_manager = camera_manager
        
        self.root.title("Student Registration System - Face Recognition")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_frame = None
        self.is_camera_running = False
        self.video_thread = None
        self.classes = {}
        self.selected_class = tk.StringVar()
        self.selected_camera = tk.StringVar()
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """Create the user interface"""
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#2196F3', height=60)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="📸 Student Registration System", 
            font=("Arial", 20, "bold"),
            bg='#2196F3', 
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Main Container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left Panel - Video Feed
        left_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        video_header = tk.Label(
            left_panel, 
            text="Camera Feed", 
            font=("Arial", 14, "bold"),
            bg='white', 
            pady=10
        )
        video_header.pack()
        
        self.video_label = tk.Label(left_panel, bg='black')
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Camera Controls
        camera_control_frame = tk.Frame(left_panel, bg='white')
        camera_control_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(camera_control_frame, text="Camera:", bg='white', font=("Arial", 10)).pack(side='left', padx=5)
        
        self.camera_dropdown = ttk.Combobox(
            camera_control_frame, 
            textvariable=self.selected_camera,
            state='readonly',
            width=25
        )
        self.camera_dropdown.pack(side='left', padx=5)
        
        self.camera_btn = tk.Button(
            camera_control_frame,
            text="▶ Start Camera",
            command=self.toggle_camera,
            bg='#4CAF50',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.camera_btn.pack(side='left', padx=5)
        
        # Status Label
        self.status_label = tk.Label(
            left_panel,
            text="Camera: Stopped",
            font=("Arial", 9),
            bg='white',
            fg='#666'
        )
        self.status_label.pack(pady=5)
        
        # Right Panel - Registration Form
        right_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2, width=400)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        
        form_header = tk.Label(
            right_panel,
            text="Student Registration Form",
            font=("Arial", 14, "bold"),
            bg='white',
            pady=15
        )
        form_header.pack()
        
        # Form Fields
        form_container = tk.Frame(right_panel, bg='white')
        form_container.pack(fill='both', expand=True, padx=20)
        
        # PRN Number
        self.create_form_field(form_container, "PRN Number:", "prn_entry", "e.g., F22113001")
        
        # Roll Number
        self.create_form_field(form_container, "Roll Number:", "roll_entry", "e.g., 101")
        
        # Full Name
        self.create_form_field(form_container, "Full Name:", "name_entry", "e.g., John Doe")
        
        # Email
        self.create_form_field(form_container, "Email:", "email_entry", "e.g., john@example.com")
        
        # Class Dropdown
        tk.Label(
            form_container, 
            text="Class:", 
            font=("Arial", 10, "bold"),
            bg='white'
        ).pack(anchor='w', pady=(15, 5))
        
        self.class_dropdown = ttk.Combobox(
            form_container,
            textvariable=self.selected_class,
            state='readonly',
            font=("Arial", 10)
        )
        self.class_dropdown.pack(fill='x', pady=(0, 15))
        
        # Register Button
        self.register_btn = tk.Button(
            form_container,
            text="📷 Capture & Register Student",
            command=self.register_student,
            bg='#2196F3',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            pady=15,
            cursor='hand2',
            state='disabled'
        )
        self.register_btn.pack(fill='x', pady=(20, 10))
        
        # Clear Button
        clear_btn = tk.Button(
            form_container,
            text="Clear Form",
            command=self.clear_form,
            bg='#f0f0f0',
            font=("Arial", 10),
            relief='flat',
            pady=10,
            cursor='hand2'
        )
        clear_btn.pack(fill='x')
        
    def create_form_field(self, parent, label_text, entry_name, placeholder):
        """Create a form field with label and entry"""
        tk.Label(
            parent, 
            text=label_text, 
            font=("Arial", 10, "bold"),
            bg='white'
        ).pack(anchor='w', pady=(15, 5))
        
        entry = tk.Entry(parent, font=("Arial", 10), relief='solid', bd=1)
        entry.pack(fill='x', ipady=5, pady=(0, 5))
        entry.insert(0, placeholder)
        entry.config(fg='grey')
        
        # Placeholder behavior
        entry.bind('<FocusIn>', lambda e: self.on_entry_focus_in(e, placeholder))
        entry.bind('<FocusOut>', lambda e: self.on_entry_focus_out(e, placeholder))
        
        setattr(self, entry_name, entry)
        
    def on_entry_focus_in(self, event, placeholder):
        if event.widget.get() == placeholder:
            event.widget.delete(0, 'end')
            event.widget.config(fg='black')
            
    def on_entry_focus_out(self, event, placeholder):
        if event.widget.get() == '':
            event.widget.insert(0, placeholder)
            event.widget.config(fg='grey')
    
    def load_data(self):
        """Load classes and cameras"""
        try:
            self.classes = self.db.get_all_classes()
            if self.classes:
                self.class_dropdown['values'] = list(self.classes.keys())
                self.class_dropdown.current(0)
            else:
                messagebox.showwarning("Warning", "No classes found in database")
                
            cameras = self.camera_manager.get_camera_list()
            if cameras:
                self.camera_dropdown['values'] = cameras
                self.camera_dropdown.current(0)
            else:
                messagebox.showerror("Error", "No cameras detected")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_camera_running:
            camera_index = self.camera_dropdown.current()
            if camera_index < 0:
                messagebox.showerror("Error", "Please select a camera")
                return
                
            self.is_camera_running = True
            self.camera_btn.config(text="⏸ Stop Camera", bg='#F44336')
            self.register_btn.config(state='normal')
            self.status_label.config(text="Camera: Running", fg='#4CAF50')
            
            self.video_thread = threading.Thread(target=self.video_loop, args=(camera_index,), daemon=True)
            self.video_thread.start()
        else:
            self.is_camera_running = False
            self.camera_btn.config(text="▶ Start Camera", bg='#4CAF50')
            self.register_btn.config(state='disabled')
            self.status_label.config(text="Camera: Stopped", fg='#666')
            
    def video_loop(self, camera_index):
        """Video processing loop"""
        cap = self.camera_manager.open_camera(camera_index)
        if not cap:
            messagebox.showerror("Error", "Failed to open camera")
            self.is_camera_running = False
            return
        
        while self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Enhance image quality
            frame = self.face_engine.enhance_image_quality(frame)
            self.current_frame = frame.copy()
            
            # Detect faces for preview
            face_locations, _ = self.face_engine.detect_and_encode_face(frame, for_registration=False)
            
            # Draw rectangles
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            self.display_frame(frame)
        
        cap.release()
    
    def display_frame(self, frame):
        """Display frame in Tkinter label"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def get_entry_value(self, entry, placeholder):
        """Get entry value, ignoring placeholder"""
        value = entry.get().strip()
        return '' if value == placeholder else value
    
    def register_student(self):
        """Register a new student"""
        # Get form values
        prn = self.get_entry_value(self.prn_entry, "e.g., F22113001")
        roll_no = self.get_entry_value(self.roll_entry, "e.g., 101")
        name = self.get_entry_value(self.name_entry, "e.g., John Doe")
        email = self.get_entry_value(self.email_entry, "e.g., john@example.com")
        class_name = self.selected_class.get()
        
        # Validate
        if not all([prn, roll_no, name, class_name]):
            messagebox.showerror("Error", "Please fill all required fields (PRN, Roll No, Name, Class)")
            return
        
        if not self.current_frame is not None:
            messagebox.showerror("Error", "No camera frame available")
            return
        
        # Process face
        try:
            face_locations, face_encodings = self.face_engine.detect_and_encode_face(
                self.current_frame, 
                for_registration=True
            )
            
            if len(face_locations) == 0:
                messagebox.showerror("Error", "No face detected. Please look at the camera.")
                return
            
            if len(face_locations) > 1:
                messagebox.showerror("Error", "Multiple faces detected. Only one person should be in frame.")
                return
            
            # Register in database
            class_id = self.classes[class_name]
            success, message = self.db.register_student(
                prn, class_id, roll_no, name, email, face_encodings[0]
            )
            
            if success:
                messagebox.showinfo("Success", f"Student registered successfully!\n\nName: {name}\nPRN: {prn}")
                self.clear_form()
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {e}")
    
    def clear_form(self):
        """Clear all form fields"""
        self.prn_entry.delete(0, 'end')
        self.prn_entry.insert(0, "e.g., F22113001")
        self.prn_entry.config(fg='grey')
        
        self.roll_entry.delete(0, 'end')
        self.roll_entry.insert(0, "e.g., 101")
        self.roll_entry.config(fg='grey')
        
        self.name_entry.delete(0, 'end')
        self.name_entry.insert(0, "e.g., John Doe")
        self.name_entry.config(fg='grey')
        
        self.email_entry.delete(0, 'end')
        self.email_entry.insert(0, "e.g., john@example.com")
        self.email_entry.config(fg='grey')
    
    def on_close(self):
        """Handle window close"""
        self.is_camera_running = False
        if self.video_thread:
            self.video_thread.join(timeout=1)
        self.root.destroy()


# ===================================================================
# Run Registration Application
if __name__ == "__main__":
    from attendance_config import DB_CONFIG, FACE_RECOGNITION_CONFIG, CAMERA_CONFIG
    
    # Initialize managers
    db = DatabaseManager(DB_CONFIG)
    face_engine = FaceRecognitionEngine(FACE_RECOGNITION_CONFIG)
    camera_mgr = CameraManager(CAMERA_CONFIG)
    
    # Create and run application
    root = tk.Tk()
    app = RegistrationApp(root, db, face_engine, camera_mgr)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    
    # Cleanup
    db.close()
