# train.py (improved - better model for higher R²)
import tensorflow as tf
from load_data import load_all_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ── Load data ─────────────────────────────────────────────
X, y = load_all_data()
print("📐 Data shape:", X.shape)

# ── Proper train/test split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Improved Model ────────────────────────────────────────
model = tf.keras.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                           input_shape=(128, 128, 6), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Block 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Block 3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),

    # Dense
    tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='huber',       # better than MSE for regression
    metrics=['mae']
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3
    )
]

# ── Train ─────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# ── Save model ────────────────────────────────────────────
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(base, "final_model.h5")
model.save(save_path)
print("✅ Model saved to:", save_path)

# ── Evaluation ────────────────────────────────────────────
preds = model.predict(X_test).squeeze()

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)

print("\n📊 MODEL EVALUATION (on test set)")
print(f"   MSE:      {mse:.4f}")
print(f"   MAE:      {mae:.4f}")
print(f"   R² Score: {r2:.4f}  ")

# ── Graphs ────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, preds, alpha=0.4, color='green')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect fit')
plt.xlabel('Actual Efficiency')
plt.ylabel('Predicted Efficiency')
plt.title(f'Actual vs Predicted  (R²={r2:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig("training_graph.png")
plt.savefig("prediction_graph.png")
print("📈 Graphs saved!")