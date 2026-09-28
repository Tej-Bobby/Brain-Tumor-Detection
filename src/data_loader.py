import tensorflow as tf

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("Classes:", train_dataset.class_names)


# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1)
])


# Normalize pixel values from 0-255 to 0-1
normalization = tf.keras.applications.mobilenet_v2.preprocess_input


# Apply augmentation and normalization to training data
train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization(data_augmentation(images, training=True)),
        labels
    )
)


# Apply only normalization to validation data
val_dataset = val_dataset.map(
    lambda images, labels: (
        normalization(images),
        labels
    )
)


# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)


# Check the final data
for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print(
        "Pixel range:",
        images.numpy().min(),
        "to",
        images.numpy().max()
    )
    print("First label:", labels[0].numpy())