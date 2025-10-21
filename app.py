from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import base64
import os
import face_recognition
import pickle

app = Flask(__name__)
CORS(app)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "mask_detector.h5")
known_faces_dir = os.path.join(base_dir, "known_faces")
encodings_file = os.path.join(base_dir, "face_encodings.pkl")

# Global variables for face recognition
known_face_encodings = []
known_face_names = []

print("\n" + "="*60)
print("STARTING FACE MASK DETECTION + RECOGNITION SYSTEM")
print("="*60)

# Check and create known_faces directory
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)
    print(f"Created directory: {known_faces_dir}")
    print("Please add face images (PersonName.jpg) to this folder!")
else:
    print(f"Known faces directory: {known_faces_dir}")

# List files in known_faces
files_in_dir = os.listdir(known_faces_dir) if os.path.exists(known_faces_dir) else []
image_files = [f for f in files_in_dir if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} image files in known_faces folder")
if image_files:
    print(f"Files: {image_files}")

# Load models
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

print("\nLoading mask detection model...")
model = load_model(model_path)
print("Mask detector loaded!")

prototxt_path = os.path.join(base_dir, "face_detector", "deploy.prototxt")
weights_path = os.path.join(base_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNet(prototxt_path, weights_path)
print("Face detector loaded!")

def load_known_faces():
    """Load and encode known faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    
    print("\n" + "-"*60)
    print("LOADING FACE RECOGNITION DATABASE")
    print("-"*60)
    
    # Try loading from pickle file first
    if os.path.exists(encodings_file):
        print("Found existing encodings file, loading...")
        try:
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_names = data['names']
                print(f"Successfully loaded {len(known_face_names)} face encodings")
                print(f"Registered people: {', '.join(known_face_names)}")
                print("-"*60)
                return
        except Exception as e:
            print(f"Error loading encodings file: {e}")
            print("Will create new encodings...")
    
    # Create new encodings from images
    if not os.path.exists(known_faces_dir):
        print(f"Known faces directory not found!")
        print("-"*60)
        return
    
    image_files = [f for f in os.listdir(known_faces_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No images found in known_faces folder!")
        print("Add images named as 'PersonName.jpg' and restart the server")
        print("-"*60)
        return
    
    print(f"Encoding {len(image_files)} face images...")
    print()
    
    for filename in image_files:
        image_path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0].replace('_', ' ')
        
        try:
            print(f"Processing: {filename}...", end=" ")
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"SUCCESS - Encoded as '{name}'")
            else:
                print(f"FAILED - No face detected in image")
        except Exception as e:
            print(f"ERROR - {str(e)}")
    
    print()
    
    # Save encodings to file
    if len(known_face_encodings) > 0:
        try:
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': known_face_encodings,
                    'names': known_face_names
                }, f)
            print(f"Saved {len(known_face_names)} face encodings to file")
            print(f"Registered people: {', '.join(known_face_names)}")
        except Exception as e:
            print(f"Warning: Could not save encodings file: {e}")
    else:
        print("No faces were successfully encoded!")
    
    print("-"*60)

# Load known faces on startup
load_known_faces()

def detect_faces(image):
    """Detect faces using OpenCV DNN"""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locations = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            
            face = image[startY:endY, startX:endX]
            
            if face.size > 0:
                faces.append(face)
                locations.append((startX, startY, endX, endY))
    
    return faces, locations

def recognize_face(face_image):
    """Recognize a face and return name + confidence"""
    try:
        # Convert to RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_face)
        
        if len(face_encodings) == 0:
            return "Unknown", 0.0
        
        if len(known_face_encodings) == 0:
            return "Unknown", 0.0
        
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            face_encoding, 
            tolerance=0.6
        )
        face_distances = face_recognition.face_distance(
            known_face_encodings, 
            face_encoding
        )
        
        name = "Unknown"
        confidence = 0.0
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1.0 - face_distances[best_match_index]
                print(f"Recognized: {name} (confidence: {confidence:.2f})")
        
        return name, float(confidence)
        
    except Exception as e:
        print(f"Face recognition error: {e}")
        return "Unknown", 0.0

def prepare_face_for_model(face):
    """Prepare face for mask detection"""
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces, locations = detect_faces(img)
        
        if len(faces) == 0:
            return jsonify({
                "no_face": True,
                "message": "No face detected in frame"
            })
        
        results = []
        for face, loc in zip(faces, locations):
            person_name, recognition_confidence = recognize_face(face)
            processed_face = prepare_face_for_model(face)
            prediction = model.predict(processed_face, verbose=0)[0]
            
            mask_prob = float(prediction[0])
            no_mask_prob = float(prediction[1])
            
            # ADD THIS DEBUG LINE
            print(f"DEBUG - Mask: {mask_prob:.3f}, No Mask: {no_mask_prob:.3f}")
            
            location_list = [int(x) for x in loc]
            
            result = {
                "location": location_list,
                "name": person_name,
                "recognition_confidence": float(recognition_confidence),
                "mask": bool(mask_prob > no_mask_prob),
                "mask_confidence": float(mask_prob),
                "no_mask_confidence": float(no_mask_prob),
                "confidence": float(max(mask_prob, no_mask_prob))
            }
            
            results.append(result)
        
        return jsonify({
            "faces_detected": int(len(faces)),
            "results": results
        })
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")  # ADD THIS
        import traceback
        traceback.print_exc()  # ADD THIS - shows full error
        return jsonify({"error": str(e)})
@app.route("/")
def home():
    return render_template("app.html")

@app.route("/status")
def status():
    """Return system status"""
    return jsonify({
        "registered_people": known_face_names,
        "total_registered": len(known_face_names)
    })

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("SERVER READY")
        print("="*60)
        print(f"Registered people: {len(known_face_names)}")
        if len(known_face_names) > 0:
            print(f"Names: {', '.join(known_face_names)}")
        print(f"\nOpen browser: http://127.0.0.1:5000")
        print("="*60 + "\n")
        
        app.run(debug=True, use_reloader=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nSERVER CRASHED: {e}")
        import traceback
        traceback.print_exc()