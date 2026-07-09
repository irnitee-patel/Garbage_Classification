import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
##dataset path
dataset_dir = "dataset"

IMG_SIZE = (128, 128)

BATCH_SIZE = 32
##data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20,
    rotation_range=20,
    zoom_range=0.20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)
##load dataset
train_data = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
print(train_data.class_indices)
##build cnn model
model = Sequential([

    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.30),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.50),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(5, activation="softmax")

])
##compile model
model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()
# ==========================
# CALLBACKS
# ==========================

os.makedirs("model", exist_ok=True)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "model/garbage_classifier.keras",

    monitor="val_accuracy",

    save_best_only=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=1e-5

)
# ==========================
# TRAIN MODEL
# ==========================

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=20,

    callbacks=[

        early_stop,

        checkpoint,

        reduce_lr

    ]

)