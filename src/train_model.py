import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping  # noqa
import os

# 📌 Path to Dataset
DATASET_PATH = "C:/Users/HP/OneDrive/Desktop/Sign_Language_Recognition/dataset"

# 📌 Training Parameters
IMG_SIZE = 64  # Increased for better feature extraction
BATCH_SIZE = 64  # Increased batch size for faster training
EPOCHS = 10  # Reduce if training takes too long

# ✅ Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# ✅ Load Data Efficiently
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ Get Number of Classes
NUM_CLASSES = len(train_data.class_indices)
print(f"✅ Dataset contains {NUM_CLASSES} classes: {list(train_data.class_indices.keys())}")

# ✅ Load EfficientNetB0 Model (Pretrained on ImageNet)
base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model

# ✅ Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),  # Reduce overfitting
    Dense(NUM_CLASSES, activation="softmax")
])

# ✅ Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Early Stopping (Stops if validation accuracy stops improving)
early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

# ✅ Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ✅ Save Model
MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/Sign_Language_Recognition/model/sign_language_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model training completed and saved at {MODEL_PATH}")
