from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit,
                             QTableWidget, QTableWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from .face_recognition import FaceRecognitionSystem
from ..utils.database import db
import pickle
import base64
from datetime import datetime

class MainWindow(QMainWindow):
    def __init__(self, camera_id: int = 0, confidence_threshold: float = 0.6):
        super().__init__()
        self.setWindowTitle("Smart Security System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize face recognition system
        self.face_system = FaceRecognitionSystem(tolerance=confidence_threshold)
        self.camera_id = camera_id
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create left panel for camera feed and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.camera_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.register_button = QPushButton("Register New User")
        self.register_button.clicked.connect(self.show_register_dialog)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.register_button)
        left_layout.addLayout(button_layout)
        
        # Add left panel to main layout
        layout.addWidget(left_panel)
        
        # Create right panel for access logs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Access logs table
        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(4)
        self.logs_table.setHorizontalHeaderLabels(["Time", "User ID", "Access Type", "Status"])
        right_layout.addWidget(self.logs_table)
        
        # Refresh logs button
        self.refresh_button = QPushButton("Refresh Logs")
        self.refresh_button.clicked.connect(self.update_logs)
        right_layout.addWidget(self.refresh_button)
        
        # Add right panel to main layout
        layout.addWidget(right_panel)
        
        # Initialize camera
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Update logs initially
        self.update_logs()

    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera!")
                return
            self.start_button.setText("Stop Camera")
            self.timer.start(30)  # Update every 30ms
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.start_button.setText("Start Camera")
            self.camera_label.clear()

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame with face recognition
            results = self.face_system.process_frame(frame)
            
            # Draw results on frame
            frame = self.face_system.draw_results(frame, results)
            
            # Log access for identified faces
            for user_id, _, confidence in results:
                if confidence > 0.6:  # Confidence threshold
                    db.log_access(user_id, "entry", confidence, "success")
            
            # Convert frame to QImage and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def show_register_dialog(self):
        dialog = RegisterDialog(self)
        if dialog.exec_():
            name = dialog.name_input.text()
            image_path = dialog.image_path
            
            if name and image_path:
                # Encode face
                face_encoding = self.face_system.encode_face(image_path)
                if face_encoding is not None:
                    # Convert numpy array to string for storage
                    face_encoding_str = base64.b64encode(
                        pickle.dumps(face_encoding)
                    ).decode('utf-8')
                    
                    # Add user to database
                    user_id = db.add_user(name, face_encoding_str)
                    if user_id:
                        QMessageBox.information(self, "Success", "User registered successfully!")
                        self.face_system.load_known_faces()  # Reload known faces
                    else:
                        QMessageBox.critical(self, "Error", "Failed to register user!")
                else:
                    QMessageBox.warning(self, "Warning", "No face detected in the image!")

    def update_logs(self):
        logs = db.get_access_logs()
        self.logs_table.setRowCount(len(logs))
        
        for i, log in enumerate(logs):
            self.logs_table.setItem(i, 0, QTableWidgetItem(
                log.access_time.strftime("%Y-%m-%d %H:%M:%S")))
            self.logs_table.setItem(i, 1, QTableWidgetItem(str(log.user_id)))
            self.logs_table.setItem(i, 2, QTableWidgetItem(log.access_type))
            self.logs_table.setItem(i, 3, QTableWidgetItem(log.status))

class RegisterDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Image selection
        image_layout = QHBoxLayout()
        self.image_label = QLabel("No image selected")
        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.select_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(select_button)
        layout.addLayout(image_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_name:
            self.image_path = file_name
            self.image_label.setText(file_name)

    def accept(self):
        if not self.name_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a name!")
            return
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image!")
            return
        self.close()

    def reject(self):
        self.close() 