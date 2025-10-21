"""
Script to capture and add a person to the known faces database
"""
import cv2
import os
import sys

def capture_face(person_name):
    """Capture a face from webcam and save it"""
    
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    
    output_path = os.path.join(known_faces_dir, f"{person_name}.jpg")
    
    if os.path.exists(output_path):
        overwrite = input(f"⚠️  {person_name} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Cancelled.")
            return False
    
    print(f"\n📷 Starting camera to capture {person_name}'s face...")
    print("👉 Look directly at the camera")
    print("👉 Ensure good lighting")
    print("👉 Remove any masks or sunglasses")
    print("👉 Press SPACE to capture, ESC to cancel\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access camera")
        return False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from camera")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected - position yourself in frame", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(f'Capture Face - {person_name}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press SPACE to capture
        if key == ord(' '):
            if len(faces) > 0:
                cv2.imwrite(output_path, frame)
                print(f"✅ Face captured and saved as: {output_path}")
                break
            else:
                print("⚠️  No face detected. Please position yourself properly.")
        
        # Press ESC to cancel
        elif key == 27:
            print("❌ Cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ {person_name} has been added to the database!")
    print("ℹ️  Restart the Flask server to load the new face.")
    return True

if __name__ == "__main__":
    print("="*50)
    print("   Add Person to Face Recognition Database")
    print("="*50)
    
    if len(sys.argv) > 1:
        person_name = " ".join(sys.argv[1:])
    else:
        person_name = input("\n👤 Enter person's name: ").strip()
    
    if not person_name:
        print("❌ Error: Name cannot be empty")
        exit(1)
    
    # Clean the name (remove special characters)
    person_name = person_name.replace("/", "_").replace("\\", "_")
    
    success = capture_face(person_name)
    
    if success:
        print("\n📋 Next steps:")
        print("1. Restart the Flask server (python app.py)")
        print("2. The system will automatically encode the new face")
        print("3. Start detection and test recognition!")