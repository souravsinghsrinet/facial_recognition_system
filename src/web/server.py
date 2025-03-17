import os
import cv2
import time
import base64
import pickle
import threading
import numpy as np
from datetime import datetime
import face_recognition
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_socketio import SocketIO

from src.utils.database import db, User, AccessLog

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'security_system_secret'
socketio = SocketIO(app)

# Global variables
camera = None
face_recognition_thread = None
thread_running = False
frame_count = 0
last_face_locations = []
last_face_names = []
last_face_confidences = []
confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
camera_error = None
# Use upload mode if camera is not available (especially for macOS Docker)
use_upload_mode = os.getenv('USE_UPLOAD_MODE', 'false').lower() == 'true'

class FaceRecognitionSystem:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from the database."""
        print("Loading known faces from database...")
        session = db.get_session()
        try:
            users = session.query(User).all()
            for user in users:
                try:
                    # Decode the string representation back to a numpy array
                    face_encoding_bytes = base64.b64decode(user.face_encoding)
                    face_encoding = pickle.loads(face_encoding_bytes)
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(user.name)
                    self.known_face_ids.append(user.id)
                    print(f"Loaded face for user: {user.name} (ID: {user.id})")
                except Exception as e:
                    print(f"Error loading face for user {user.name}: {e}")
        finally:
            session.close()
            
        print(f"Loaded {len(self.known_face_encodings)} faces from database")

    def identify_faces(self, frame):
        """Identify faces in the given frame."""
        # Only process every other frame to save CPU
        global frame_count, last_face_locations, last_face_names, last_face_confidences
        frame_count += 1
        if frame_count % 3 != 0:
            return last_face_locations, last_face_names, last_face_confidences
            
        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_ids = []
        confidences = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for any known face
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            user_id = None
            confidence = 0.0
            
            if len(self.known_face_encodings) > 0:
                # Calculate face distances
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]
                    
                    if matches[best_match_index] and confidence >= self.confidence_threshold:
                        name = self.known_face_names[best_match_index]
                        user_id = self.known_face_ids[best_match_index]
                        
                        # Log access
                        db.log_access(
                            user_id=user_id,
                            access_type="entry",
                            confidence=float(confidence),
                            status="success"
                        )
                        print(f"Access granted to {name} (confidence: {confidence:.2f})")
                        
                        # Send notification via SocketIO
                        socketio.emit('access_granted', {
                            'name': name,
                            'confidence': float(confidence),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        print(f"Access denied: Unknown person or low confidence ({confidence:.2f})")
                        
                        # Send notification via SocketIO
                        socketio.emit('access_denied', {
                            'confidence': float(confidence),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            
            face_names.append(name)
            face_ids.append(user_id)
            confidences.append(confidence)
        
        # Scale face locations back to original size
        face_locations = [(top*4, right*4, bottom*4, left*4) 
                         for (top, right, bottom, left) in face_locations]
        
        last_face_locations = face_locations
        last_face_names = face_names
        last_face_confidences = confidences
        
        return face_locations, face_names, confidences

    def process_uploaded_image(self, image_data):
        """Process an uploaded image for face recognition."""
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert from BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        face_ids = []
        confidences = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for any known face
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            user_id = None
            confidence = 0.0
            
            if len(self.known_face_encodings) > 0:
                # Calculate face distances
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]
                    
                    if matches[best_match_index] and confidence >= self.confidence_threshold:
                        name = self.known_face_names[best_match_index]
                        user_id = self.known_face_ids[best_match_index]
                        
                        # Log access
                        db.log_access(
                            user_id=user_id,
                            access_type="entry",
                            confidence=float(confidence),
                            status="success"
                        )
                        print(f"Access granted to {name} (confidence: {confidence:.2f})")
                        
                        # Send notification via SocketIO
                        socketio.emit('access_granted', {
                            'name': name,
                            'confidence': float(confidence),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        print(f"Access denied: Unknown person or low confidence ({confidence:.2f})")
                        
                        # Send notification via SocketIO
                        socketio.emit('access_denied', {
                            'confidence': float(confidence),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            
            face_names.append(name)
            face_ids.append(user_id)
            confidences.append(confidence)
        
        # Draw results on the frame
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'processed_image': processed_image,
            'face_count': len(face_locations),
            'recognized_names': face_names,
            'confidences': [float(c) for c in confidences]
        }

def get_camera(camera_id=0):
    """Get a camera instance."""
    global camera, camera_error, use_upload_mode
    
    if use_upload_mode:
        camera_error = "Camera access disabled. Using upload mode instead."
        return None
        
    if camera is None:
        try:
            camera_id = int(camera_id)
            camera = cv2.VideoCapture(camera_id)
            if not camera.isOpened():
                camera_error = f"Error: Could not open camera {camera_id}. Check if your camera is connected and not used by another application."
                print(camera_error)
                return None
            else:
                # Reset error if camera opens successfully
                camera_error = None
        except Exception as e:
            camera_error = f"Error accessing camera: {str(e)}"
            print(camera_error)
            return None
    return camera

def generate_frames():
    """Generate video frames with face recognition."""
    global camera_error, use_upload_mode
    face_rec = FaceRecognitionSystem(confidence_threshold=confidence_threshold)
    
    if use_upload_mode:
        # In upload mode, show a message to use the upload feature
        while True:
            # Create an info image with text
            info_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(info_img, "Upload Mode Active", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(info_img, "Use the 'Upload Image' button below", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, "for face recognition", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', info_img)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)  # Delay to avoid flooding
    
    camera = get_camera(int(os.getenv('CAMERA_ID', '0')))
    
    if camera is None:
        # If camera can't be accessed, yield an error image
        while True:
            # Create an error image with text
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera Error:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_img, camera_error or "Could not access camera", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_img, "Try using Upload Mode instead", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)  # Delay to avoid flooding with error images
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame")
            break
        
        # Identify faces in the frame
        face_locations, face_names, confidences = face_rec.identify_faces(frame)
        
        # Draw results on the frame
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', confidence_threshold=confidence_threshold, 
                          camera_error=camera_error, use_upload_mode=use_upload_mode)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/users')
def users():
    """Display all users."""
    session = db.get_session()
    try:
        users_list = session.query(User).all()
        return render_template('users.html', users=users_list)
    finally:
        session.close()

@app.route('/access_logs')
def access_logs():
    """Display access logs."""
    session = db.get_session()
    try:
        # Join access logs with users to get user names
        logs = session.query(
            AccessLog, User.name
        ).join(
            User, AccessLog.user_id == User.id
        ).order_by(
            AccessLog.access_time.desc()
        ).limit(50).all()
        
        return render_template('access_logs.html', logs=logs)
    finally:
        session.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new user with face recognition."""
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return render_template('register.html', error="Please enter a name.")
        
        image_source = request.form.get('image_source', 'webcam')
        image_data = None
        
        # Process based on image source
        if image_source == 'webcam':
            # Get image from webcam (base64 encoded)
            webcam_image = request.form.get('webcam_image')
            if webcam_image:
                # Remove header from base64 string
                image_data = webcam_image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                return render_template('register.html', error="Please capture an image from the webcam.")
        else:  # image_source == 'upload'
            # Get uploaded image file
            uploaded_file = request.files.get('upload_image')
            if uploaded_file:
                image_bytes = uploaded_file.read()
            else:
                return render_template('register.html', error="Please select an image file to upload.")
        
        if image_bytes:
            try:
                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return render_template('register.html', error="Could not process the image. Please try again with a different image.")
                
                # Convert from BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find faces in the image
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    return render_template('register.html', error="No face detected in the image. Please try again with a clearer photo.")
                
                # Get the first face encoding
                face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                
                # Convert the numpy array to a string for storage
                face_encoding_bytes = pickle.dumps(face_encoding)
                face_encoding_str = base64.b64encode(face_encoding_bytes).decode('utf-8')
                
                # Add the user to the database
                user_id = db.add_user(name, face_encoding_str)
                
                if user_id:
                    return redirect(url_for('users'))
                else:
                    return render_template('register.html', error="Failed to add user to database.")
            except Exception as e:
                print(f"Error processing image: {e}")
                return render_template('register.html', error=f"Error processing image: {str(e)}")
        else:
            return render_template('register.html', error="No image data received. Please try again.")
    
    return render_template('register.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Process an uploaded image for face recognition."""
    # Get image data from the request
    image_data = request.files.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400
        
    # Read the image file
    image_bytes = image_data.read()
    
    # Process the image
    face_rec = FaceRecognitionSystem(confidence_threshold=confidence_threshold)
    result = face_rec.process_uploaded_image(image_bytes)
    
    return jsonify(result)

def start_server():
    """Start the Flask server."""
    app.debug = False
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    print(f"Starting server on {host}:{port}")
    if use_upload_mode:
        print("Running in UPLOAD MODE - camera access is disabled")
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)

# Run the app if executed directly
if __name__ == '__main__':
    start_server() 