#!/usr/bin/env python3
"""
Train a small CNN to classify individual digit images (0-9).
Uses digit crops extracted by extract_digits.py.
"""

import numpy as np
import os
import sys
import cv2
from pathlib import Path


def load_digit_data(data_dir):
    """Load digit images from data_dir/{0..9}/ directories."""
    images = []
    labels = []
    
    for digit in range(10):
        digit_dir = Path(data_dir) / str(digit)
        if not digit_dir.exists():
            print(f"  Warning: no samples for digit {digit}")
            continue
        
        for img_path in sorted(digit_dir.glob('*.png')):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(digit)
    
    return np.array(images), np.array(labels)


def augment_image(img, rng):
    """Apply random augmentation to a digit image."""
    h, w = img.shape
    
    # Random rotation (±8 degrees)
    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    # Random shift (±2 pixels)
    dx = rng.integers(-2, 3)
    dy = rng.integers(-2, 3)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    # Random brightness adjustment
    factor = rng.uniform(0.7, 1.3)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    return img


def build_model():
    """Build a small CNN for digit classification."""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Conv block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    data_dir = Path('data/digits')
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    print("Loading digit data...")
    images, labels = load_digit_data(data_dir)
    
    if len(images) == 0:
        print("No digit data found! Run extract_digits.py first.")
        sys.exit(1)
    
    print(f"Loaded {len(images)} digit images")
    for d in range(10):
        count = np.sum(labels == d)
        if count > 0:
            print(f"  Digit {d}: {count} samples")
    
    # Augment data to increase training set size (especially for underrepresented digits)
    print("\nAugmenting data...")
    rng = np.random.default_rng(42)
    
    aug_images = list(images)
    aug_labels = list(labels)
    
    # Target at least 30 samples per digit class
    target_per_class = 30
    for d in range(10):
        class_imgs = images[labels == d]
        current = len(class_imgs)
        if current == 0:
            continue
        
        needed = max(0, target_per_class - current)
        for _ in range(needed):
            idx = rng.integers(0, current)
            aug_img = augment_image(class_imgs[idx], rng)
            aug_images.append(aug_img)
            aug_labels.append(d)
    
    # Also augment everything 3x for more variety
    for img, lbl in zip(images, labels):
        for _ in range(3):
            aug_images.append(augment_image(img, rng))
            aug_labels.append(lbl)
    
    X = np.array(aug_images, dtype=np.float32) / 255.0
    y = np.array(aug_labels)
    
    # Add channel dimension
    X = X.reshape(-1, 28, 28, 1)
    
    print(f"Training set: {len(X)} samples (after augmentation)")
    
    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    
    # Split 85/15 train/val
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Build and train
    print("\nTraining CNN...")
    model = build_model()
    model.summary()
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    model_path = str(model_dir / 'digit_cnn.keras')
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    
    # Save in both formats
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Also save as TFLite for lightweight deployment
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = str(model_dir / 'digit_cnn.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"TFLite conversion skipped: {e}")


if __name__ == '__main__':
    main()
