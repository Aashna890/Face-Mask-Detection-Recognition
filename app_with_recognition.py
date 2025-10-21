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

# Load mask detection model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "mask_detector.h5")

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found")
    exit(1)

model = load_model(model_path)
print("Mask detector loaded!")

# Load face detector
prototxt_path = os.path.join(base_dir, "face_detector", "deploy.prototxt")
weights_path = os.path.join(base_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNet(prototxt_path, weights_path)
print("Face detector loaded!")

# Face recognition setup
known_faces_dir = os.path.join(base_dir, "known_faces")
encodings_file = os.path.join(base_dir, "face_encodings.pkl")
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    
    if os.path.exists(encodings_file):
        print("Loading face encodings from file...")
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
            print(f"Loaded {len(known_face_names)} known faces: {known_face_names}")
        return
    
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created known_faces directory. Add photos and restart.")
        return
    
    print("Encoding known faces...")
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            name = os.path.splitext(filename)[0]
            
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"  ✓ Encoded: {name}")
                else:
                    print(f"  ✗ No face in: {filename}")
            except Exception as e:
                print(f"  ✗ Error: {filename} - {e}")
    
    if len(known_face_encodings) > 0:
        with open(encodings_file, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
        print(f"Saved {len(known_face_names)} encodings")

load_known_faces()

def detect_faces(image):
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
    try:
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        
        if len(face_encodings) == 0 or len(known_face_encodings) == 0:
            return "Unknown", 0.0
        
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Unknown"
        confidence = 0.0
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1.0 - face_distances[best_match_index]
        
        return name, float(confidence)
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", 0.0

def prepare_face_for_model(face):
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
            return jsonify({"no_face": True, "message": "No face detected"})
        
        results = []
        for face, loc in zip(faces, locations):
            person_name, recognition_confidence = recognize_face(face)
            processed_face = prepare_face_for_model(face)
            prediction = model.predict(processed_face, verbose=0)[0]
            
            mask_prob = float(prediction[0])
            no_mask_prob = float(prediction[1])
            location_list = [int(x) for x in loc]
            
            results.append({
                "location": location_list,
                "name": person_name,
                "recognition_confidence": recognition_confidence,
                "mask": bool(mask_prob > no_mask_prob),
                "mask_confidence": mask_prob,
                "no_mask_confidence": no_mask_prob,
                "confidence": max(mask_prob, no_mask_prob)
            })
        
        return jsonify({"faces_detected": int(len(faces)), "results": results})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return render_template("app.html")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(f"Known faces: {len(known_face_names)}")
    print(f"Names: {known_face_names}")
    print("="*50 + "\n")
    app.run(debug=True)