import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_FE   = 3
EPOCHS_FT   = 5
NUM_CLASSES = 2
LR_FE       = 1e-3
LR_FT       = 1e-5

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

import pathlib, zipfile, requests, os

url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "cats_and_dogs.zip"

if not os.path.exists("cats_and_dogs_filtered"):
    print("Downloading dataset (60MB)...")
    r = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    print("Done!")

TRAIN_DIR = "cats_and_dogs_filtered/train"
VAL_DIR   = "cats_and_dogs_filtered/validation"

def preprocess(img, lbl):
    img = tf.cast(img, tf.float32) / 255.0
    return img, lbl

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

print(f"Classes: {train_ds.class_names if hasattr(train_ds, 'class_names') else 'cats, dogs'}")
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

print(f"VGG16 loaded. Total layers: {len(base_model.layers)}")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

print("\n=== PHASE 1: Feature Extraction (VGG16 frozen) ===")
model.compile(optimizer=Adam(LR_FE), loss="categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history_fe = model.fit(train_ds, epochs=EPOCHS_FE, validation_data=val_ds, callbacks=[early_stop])

print("\n=== PHASE 2: Fine-Tuning (unfreeze last conv block) ===")
for layer in base_model.layers[15:]:
    layer.trainable = True
model.compile(optimizer=Adam(LR_FT), loss="categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_ds, epochs=EPOCHS_FT, validation_data=val_ds, callbacks=[early_stop])

loss, acc = model.evaluate(val_ds)
print(f"\nValidation Accuracy: {acc:.4f}")

def plot_history(h1, h2, metric="accuracy"):
    values   = h1.history[metric] + h2.history[metric]
    val_vals = h1.history["val_"+metric] + h2.history["val_"+metric]
    epochs   = range(1, len(values)+1)
    fe_end   = len(h1.history[metric])
    plt.figure(figsize=(10,4))
    plt.plot(epochs, values,   "b-o", label=f"Train {metric}")
    plt.plot(epochs, val_vals, "r-o", label=f"Val {metric}")
    plt.axvline(x=fe_end, color="gray", linestyle="--", label="Fine-Tuning starts")
    plt.title(f"{metric.capitalize()} over Epochs")
    plt.xlabel("Epoch"); plt.ylabel(metric.capitalize()); plt.legend()
    plt.tight_layout(); plt.savefig(f"{metric}_plot.png"); plt.show()
    print(f"Saved: {metric}_plot.png")

plot_history(history_fe, history_ft, "accuracy")
plot_history(history_fe, history_ft, "loss")

model.save("vgg16_cats_dogs_finetuned.h5")
print("\nModel saved: vgg16_cats_dogs_finetuned.h5")
print("Done!")