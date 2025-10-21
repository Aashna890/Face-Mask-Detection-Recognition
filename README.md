# Face-Mask-Detection-Recognition
Real-time AI system for face mask detection and recognition using Flask, OpenCV, and TensorFlow.


## 🚀 Features
- Dual AI: Face recognition + Mask detection  
- Real-time processing (~1.5s per frame)  
- Flask + TensorFlow + OpenCV integration  
- User-friendly web interface  
- 90–95% mask detection accuracy  

---

## 🧠 Tech Stack
**Frontend:** HTML, CSS, JavaScript, WebRTC  
**Backend:** Flask, REST API, JSON, CORS  
**AI Models:** TensorFlow/Keras (MobileNetV2), OpenCV, face_recognition (dlib)

---

## ⚙️ System Architecture
![Architecture Diagram](images/architecture.png) <!-- optional -->
1. Browser captures video feed (via WebRTC)  
2. Flask backend processes frame using OpenCV  
3. AI models detect and recognize face  
4. Mask classification + identity shown on UI  

---

## 🧩 Model Details
**Model Used:** MobileNetV2 (Transfer Learning)  
- Optimizer: Adam  
- Learning rate: 0.0001  
- Loss: Binary Crossentropy  
- Epochs: 20  
- Accuracy: ~92%  
- Input size: 224×224 pixels  


```bash
git clone https://github.com/your-username/Face-Mask-Detection-Recognition.git
cd Face-Mask-Detection-Recognition
