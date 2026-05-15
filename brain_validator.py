import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset_path = "brain_validator_dataset"

img_size = 224

batch_size = 32

datagen = ImageDataGenerator(

    rescale=1. / 255,

    validation_split=0.2,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True
)

train_set = datagen.flow_from_directory(

    dataset_path,

    target_size=(img_size, img_size),

    batch_size=batch_size,

    class_mode='binary',

    subset='training',

    shuffle=True
)


val_set = datagen.flow_from_directory(

    dataset_path,

    target_size=(img_size, img_size),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation',

    shuffle=False
)

print("\nClass Labels:")

print(train_set.class_indices)

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)


base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation='relu'
)(x)

x = Dropout(0.5)(x)

x = Dense(
    128,
    activation='relu'
)(x)

x = Dropout(0.3)(x)

predictions = Dense(
    1,
    activation='sigmoid'
)(x)

model = Model(

    inputs=base_model.input,

    outputs=predictions
)

model.compile(

    optimizer=Adam(learning_rate=0.0001),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.2,

    patience=2,

    verbose=1
)

checkpoint = ModelCheckpoint(

    "best_brain_validator_model.h5",

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

history = model.fit(

    train_set,

    validation_data=val_set,

    epochs=15,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

model.save(
    "brain_validator_mobilenetv2_final.h5"
)

print("\nModel saved successfully.")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('MobileNetV2 Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('MobileNetV2 Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()

val_set.reset()

predictions = model.predict(val_set)

y_pred = (predictions > 0.5).astype(int)

y_true = val_set.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=['Brain', 'Not Brain'],

    yticklabels=['Brain', 'Not Brain']
)

plt.title('MobileNetV2 Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

print("\nClassification Report:\n")

print(classification_report(

    y_true,

    y_pred,

    target_names=['Brain', 'Not Brain']
))