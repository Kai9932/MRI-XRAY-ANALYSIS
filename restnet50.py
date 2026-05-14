# ============================================================
# FILE: resnet50.py
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = r"C:\Users\User\Desktop\WK xmum\machine learning\archive\brain_dataset"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=20,
    zoom_range=0.15,

    width_shift_range=0.1,
    height_shift_range=0.1,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2],

    validation_split=0.2
)

# ============================================================
# TRAIN SET
# ============================================================

train_set = train_datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training',

    shuffle=True
)

# ============================================================
# VALIDATION SET
# ============================================================

val_set = train_datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='validation',

    shuffle=False
)

print(train_set.class_indices)

# ============================================================
# LOAD RESNET50
# ============================================================

base_model = ResNet50(
    weights='imagenet',

    include_top=False,

    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze backbone
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),

    loss='binary_crossentropy',

    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',

    patience=4,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',

    factor=0.2,

    patience=2,

    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_resnet50_model.h5",

    monitor='val_accuracy',

    save_best_only=True
)

# ============================================================
# TRAINING
# ============================================================

history = model.fit(
    train_set,

    validation_data=val_set,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# ============================================================
# FINE TUNING
# ============================================================

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

history_finetune = model.fit(
    train_set,

    validation_data=val_set,

    epochs=5
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('ResNet50 Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('ResNet50 Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()

# ============================================================
# PREDICTIONS
# ============================================================

val_set.reset()

predictions = model.predict(val_set)

y_pred = (predictions > 0.5).astype(int)

y_true = val_set.classes

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=['No Tumor', 'Tumor'],

    yticklabels=['No Tumor', 'Tumor']
)

plt.title('ResNet50 Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(classification_report(
    y_true,

    y_pred,

    target_names=['No Tumor', 'Tumor']
))

# ============================================================
# SAVE MODEL
# ============================================================

model.save("final_hybrid_resnet50_model.h5")

print("Training Completed")
print("Model saved as final_hybrid_resnet50_model.h5")