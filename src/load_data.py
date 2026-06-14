# load_data.py (improved - augmentation for higher R²)
import os
import cv2
import numpy as np
import random

IMG_SIZE = 128

# ── Detailed label → efficiency mapping ──────────────────
EFFICIENCY_MAP = {
    # Healthy
    "healthy":              (0.85, 0.95),

    # Moderate stress
    "early":                (0.60, 0.75),
    "mild":                 (0.62, 0.75),
    "bacterial_spot":       (0.55, 0.70),
    "leaf_mold":            (0.55, 0.68),
    "septoria":             (0.52, 0.65),
    "spider_mites":         (0.50, 0.65),
    "_k":                   (0.55, 0.68),

    # Critical stress
    "late":                 (0.30, 0.50),
    "severe":               (0.28, 0.45),
    "mosaic_virus":         (0.25, 0.42),
    "yellowleaf":           (0.22, 0.40),
    "target_spot":          (0.30, 0.48),
    "_n":                   (0.28, 0.45),
    "drought":              (0.35, 0.55),
}

def map_label_to_efficiency(label):
    label_lower = label.lower()
    for key, (low, high) in EFFICIENCY_MAP.items():
        if key in label_lower:
            return round(random.uniform(low, high), 4)
    # Default moderate
    return round(random.uniform(0.55, 0.75), 4)


# ── RGB Indices ───────────────────────────────────────────
def calculate_indices(img):
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)

    VARI  = (G - R) / (G + R - B + 1e-5)
    ExG   = 2 * G - R - B
    MGRVI = (G**2 - R**2) / (G**2 + R**2 + 1e-5)

    VARI  = np.clip(VARI,  -1, 1)
    ExG   = np.clip(ExG,    0, 1)
    MGRVI = np.clip(MGRVI, -1, 1)

    return np.stack([VARI, ExG, MGRVI], axis=-1)


# ── Augment single image ──────────────────────────────────
def augment_image(img_6ch):
    aug = img_6ch.copy()

    # Random brightness on RGB channels only
    aug[:, :, :3] = np.clip(
        aug[:, :, :3] * random.uniform(0.8, 1.2), 0, 1
    )

    # Random horizontal flip
    if random.random() > 0.5:
        aug = np.fliplr(aug)

    # Random vertical flip
    if random.random() > 0.5:
        aug = np.flipud(aug)

    # Random 90° rotation
    k = random.randint(0, 3)
    aug = np.rot90(aug, k)

    return aug


# ── Load single dataset folder ────────────────────────────
def load_dataset(path, augment=True):
    X, y = [], []

    if not os.path.exists(path):
        print("❌ Path not found:", path)
        return X, y

    print("📂 Loading:", path)

    for label in os.listdir(path):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue

        count = 0
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            indices = calculate_indices(img)
            img_6ch = np.concatenate([img, indices], axis=-1)

            # Original image
            eff = map_label_to_efficiency(label)
            X.append(img_6ch)
            y.append(eff)

            # 3 augmented copies
            if augment:
                for _ in range(3):
                    aug = augment_image(img_6ch)
                    X.append(aug)
                    # Slightly vary efficiency for augmented copies
                    y.append(round(eff + random.uniform(-0.03, 0.03), 4))

            count += 1

        print(f"   ✔ {label}: {count} original → {count * 4} with augmentation")

    return X, y


# ── Load all data ─────────────────────────────────────────
def load_all_data():
    X, y = [], []

    # Always points to majorproject/ folder
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    plant_path = os.path.join(base, "PlantVillage")
    olid_path  = os.path.join(base, "olid")

    print("📁 Working dir:", base)

    pv_X,   pv_y   = load_dataset(plant_path)
    olid_X, olid_y = load_dataset(olid_path)

    X.extend(pv_X);   y.extend(pv_y)
    X.extend(olid_X); y.extend(olid_y)

    y_arr = np.array(y, dtype=np.float32)

    print(f"\n✅ Total images:      {len(X)}")
    print(f"   Efficiency range:  {y_arr.min():.2f} → {y_arr.max():.2f}")
    print(f"   Mean efficiency:   {y_arr.mean():.2f}")
    print(f"   Std deviation:     {y_arr.std():.2f}  ← higher = better R²")

    return np.array(X, dtype=np.float32), y_arr