import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    return img


def calculate_indices(img):
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]

    VARI = (G - R) / (G + R - B + 1e-5)
    ExG = 2*G - R - B
    MGRVI = (G**2 - R**2) / (G**2 + R**2 + 1e-5)

    indices = np.stack([VARI, ExG, MGRVI], axis=-1)

    return indices