# Face-Mask-Detection-Recognition
Real-time AI system for face mask detection and recognition using Flask, OpenCV, and TensorFlow.


## Overview
 
This system combines two deep learning pipelines:
 
1. **Mask Detection** — A MobileNetV2-based CNN classifies each detected face as `Mask` or `No Mask` in real time.
2. **Face Recognition** — Each detected face is matched against a database of registered individuals and labeled with their name.
Both pipelines run simultaneously on every frame captured from the webcam, with results rendered live in the browser via a Flask API.
 
---
 
## Features
 
- Real-time mask detection through browser webcam
- Face recognition with confidence scoring
- Supports multiple faces in a single frame
- Register new people via webcam capture or by dropping a photo
- Persistent face encoding database (no re-encoding on restart)
- REST API (`/detect`) for easy integration
- Training pipeline with data augmentation and accuracy/loss plots
---
 
## Tech Stack
 
| Component | Technology |
|---|---|
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Deep Learning | TensorFlow 2.10 / Keras |
| Face Detection | OpenCV DNN (SSD ResNet-10) |
| Face Recognition | `face_recognition` (dlib) |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, JavaScript, Canvas API |
| Data Processing | NumPy, scikit-learn |
| Visualization | Matplotlib |
 
---
 
## Project Structure
 
```
face-mask-recognition/
│
├── dataset/                          # Training data
│   ├── with_mask/
│   └── without_mask/
│
├── face_detector/                    # OpenCV SSD face detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── known_faces/                      # Registered face images
│   └── PersonName.jpg
│
├── templates/
│   └── app.html                      # Browser UI
│
├── app.py                            # Main Flask server
├── train_mask_detector.py            # Model training (TF/Keras)
├── detect_mask_video.py              # Standalone OpenCV webcam script
├── add_known_person.py               # Register a new face via webcam
├── test_model.py                     # Test model on a single image
├── fix_model.py                      # Fix Keras batch_shape compatibility
├── verify_setup.py                   # Check installation and file setup
├── requirements.txt
├── mask_detector.h5                  # Trained model weights (after training)
├── face_encodings.pkl                # Cached face recognition database
└── plot.png                          # Training accuracy/loss chart
```
 
---
 
## Installation
 
### Requirements
- Python 3.8 – 3.10
- Webcam
- Git
### 1. Clone
 
```bash
git clone https://github.com/your-username/face-mask-recognition.git
cd face-mask-recognition
```
 
### 2. Create virtual environment
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
> **Windows note:** `face_recognition` requires dlib. If the install fails, run:
> ```bash
> pip install cmake dlib
> pip install face_recognition
> ```
 
### 4. Download face detector weights
 
Place these two files inside `face_detector/`:
 
- [`deploy.prototxt`](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- [`res10_300x300_ssd_iter_140000.caffemodel`](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
---
 
## Usage
 
### 1. Train the mask detector
 
Download the [Face Mask Dataset from Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) and place it as:
 
```
dataset/
├── with_mask/
└── without_mask/
```
 
Then run:
 
```bash
python train_mask_detector.py
```
 
Outputs `mask_detector.h5` and `plot.png`.
 
---
 
### 2. Register faces for recognition
 
**Option A — Capture from webcam:**
```bash
python add_known_person.py
```
 
**Option B — Drop a photo manually:**
Save a clear, front-facing photo as `known_faces/PersonName.jpg` and restart the server.
 
---
 
### 3. Start the web app
 
```bash
python app.py
```
 
Open **http://127.0.0.1:5000** in your browser.
 
---
 
### 4. Standalone webcam (no browser)
 
```bash
python detect_mask_video.py
```
 
Press `Q` to quit.
 
---
 
### Verify setup
 
```bash
python verify_setup.py
```
 
---
 
## Model Architecture
 
```
Input Image (224 × 224 × 3)
          │
          ▼
  MobileNetV2 backbone
  (pretrained ImageNet weights, frozen)
          │
          ▼
  AveragePooling2D (7 × 7)
          │
          ▼
      Flatten
          │
          ▼
    Dense (128, ReLU)
          │
          ▼
      Dropout (0.5)
          │
          ▼
   Dense (2, Softmax)
          │
          ▼
  [with_mask | without_mask]
```
 
**Training config:**
 
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Binary cross-entropy |
| Epochs | 20 |
| Batch size | 32 |
| Input size | 224 × 224 |
| Augmentation | Rotation, zoom, flip, shift, shear |
 
---
 
## API Reference
 
### `POST /detect`
 
Send a base64-encoded webcam frame, receive detection results.
 
**Request body:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```
 
**Response:**
```json
{
  "faces_detected": 2,
  "results": [
    {
      "location": [x1, y1, x2, y2],
      "name": "Aashna Gaikwad",
      "recognition_confidence": 0.87,
      "mask": true,
      "mask_confidence": 0.96,
      "no_mask_confidence": 0.04,
      "confidence": 0.96
    }
  ]
}
```
 
### `GET /status`
 
Returns list of all registered people in the recognition database.
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Training accuracy | ~98% |
| Validation accuracy | ~97% |
| Face detection model | SSD ResNet-10 |
| Inference speed | Real-time (webcam) |
 
---
 
## Troubleshooting
 
**`ValueError: Unrecognized keyword arguments: ['batch_shape']`**
 
Keras version mismatch. Retrain the model:
```bash
python train_mask_detector.py
```
Or run the compatibility fix:
```bash
python fix_model.py
```
 
**CUDA warning on startup**
```
Could not load dynamic library 'cudart64_110.dll'
```
Safe to ignore — the system runs on CPU without a GPU.
 
**`face_recognition` fails to install**
```bash
pip install cmake dlib
pip install face_recognition
```
 
---
- Loss: Binary Crossentropy  
- Epochs: 20  
- Accuracy: ~92%  
- Input size: 224×224 pixels  

