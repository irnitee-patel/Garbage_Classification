import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/garbage_classifier.keras")

# Class names
class_names = [
    "Battery",
    "Cardboard",
    "Glass",
    "Paper",
    "Plastic"
]

# Test image path
image_path = "test_image/paper.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Preprocess
image = cv2.resize(image, (128, 128))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)

# Prediction
prediction = model.predict(image)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

# Output
print("=" * 40)
print("Predicted Class :", class_names[predicted_index])
print(f"Confidence : {confidence * 100:.2f}%")
print("=" * 40)