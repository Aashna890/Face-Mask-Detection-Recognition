# test_model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

model = load_model("mask_detector.h5")

# Test with your own image
test_image_path = "test_with_mask.jpg"  # Take a photo of yourself with mask
image = load_img(test_image_path, target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)[0]
print(f"With Mask: {prediction[0]:.3f}")
print(f"Without Mask: {prediction[1]:.3f}")

if prediction[0] > prediction[1]:
    print("Result: MASK DETECTED")
else:
    print("Result: NO MASK")