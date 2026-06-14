import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = 128

def calculate_indices(img):
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]

    VARI = (G - R) / (G + R - B + 1e-5)
    ExG = 2*G - R - B
    MGRVI = (G**2 - R**2) / (G**2 + R**2 + 1e-5)

    return np.stack([VARI, ExG, MGRVI], axis=-1)

# Load model safely
model = tf.keras.models.load_model("final_model.h5", compile=False)

img_path = input("Enter image path: ")

img = cv2.imread(img_path)

if img is None:
    print("❌ Invalid image path")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

indices = calculate_indices(img)
img = np.concatenate([img, indices], axis=-1)
img = np.expand_dims(img, axis=0)

# Safe prediction
pred = model.predict(img)
print("Raw prediction:", pred)

pred = float(pred.squeeze())

if pred > 0.8:
    status = "Healthy"
elif pred > 0.6:
    status = "Moderate"
else:
    status = "Critical"

print(f"Efficiency: {pred*100:.2f}%")
print("Status:", status)