"""
Verify Face Recognition Setup
"""
import os
import sys

print("\n" + "="*60)
print("FACE RECOGNITION SETUP VERIFICATION")
print("="*60 + "\n")

# Check 1: Known faces directory
known_faces_dir = "known_faces"
print(f"1. Checking '{known_faces_dir}' directory...")
if os.path.exists(known_faces_dir):
    print(f"   ✓ Directory exists")
    
    files = os.listdir(known_faces_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) > 0:
        print(f"   ✓ Found {len(image_files)} image(s):")
        for img in image_files:
            file_path = os.path.join(known_faces_dir, img)
            size = os.path.getsize(file_path)
            name = os.path.splitext(img)[0]
            print(f"      - {img} ({size} bytes) → Will be registered as '{name}'")
    else:
        print(f"   ✗ No images found!")
        print(f"   → Add images named as 'PersonName.jpg'")
else:
    print(f"   ✗ Directory not found!")
    print(f"   → Creating directory...")
    os.makedirs(known_faces_dir)
    print(f"   ✓ Directory created. Add images and restart.")

# Check 2: face_recognition library
print(f"\n2. Checking face_recognition library...")
try:
    import face_recognition
    print(f"   ✓ face_recognition is installed")
except ImportError:
    print(f"   ✗ face_recognition not installed!")
    print(f"   → Run: pip install face_recognition")
    sys.exit(1)

# Check 3: Try encoding a test face
print(f"\n3. Testing face encoding...")
if len(image_files) > 0:
    test_image = os.path.join(known_faces_dir, image_files[0])
    try:
        print(f"   Testing with: {image_files[0]}")
        image = face_recognition.load_image_file(test_image)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            print(f"   ✓ Successfully detected and encoded face!")
            print(f"   ✓ Face encoding shape: {encodings[0].shape}")
        else:
            print(f"   ✗ No face detected in image!")
            print(f"   → Make sure the image shows a clear front-facing face")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"   ⚠ Skipped (no images to test)")

# Check 4: Model files
print(f"\n4. Checking model files...")
if os.path.exists("mask_detector.h5"):
    print(f"   ✓ mask_detector.h5 found")
else:
    print(f"   ✗ mask_detector.h5 not found!")
    print(f"   → Run training script first")

if os.path.exists("face_detector/deploy.prototxt"):
    print(f"   ✓ face_detector/deploy.prototxt found")
else:
    print(f"   ✗ face_detector files missing!")

# Check 5: Template
print(f"\n5. Checking template...")
if os.path.exists("templates/app.html"):
    print(f"   ✓ templates/app.html found")
else:
    print(f"   ✗ templates/app.html not found!")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60 + "\n")

if len(image_files) > 0:
    print("✓ Setup looks good! Run: python app.py")
else:
    print("⚠ Add face images to 'known_faces' folder first!")
    print("\nHow to add faces:")
    print("1. Take/find a clear photo of each person")
    print("2. Name it as: PersonName.jpg (e.g., John_Doe.jpg)")
    print("3. Save to 'known_faces' folder")
    print("4. Run: python app.py")

print()