import os
import shutil
import random

SOURCE_DIR = "data/raw/Training"
DEST_DIR = "data/processed"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

train_ratio = 0.8

random.seed(42)

for class_name in classes:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    train_class_dir = os.path.join(DEST_DIR, "train", class_name)
    val_class_dir = os.path.join(DEST_DIR, "val", class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    images = os.listdir(source_class_dir)

    random.shuffle(images)

    train_size = int(len(images) * train_ratio)

    train_images = images[:train_size]
    val_images = images[train_size:]

    for image in train_images:
        source = os.path.join(source_class_dir, image)
        destination = os.path.join(train_class_dir, image)
        shutil.copy2(source, destination)

    for image in val_images:
        source = os.path.join(source_class_dir, image)
        destination = os.path.join(val_class_dir, image)
        shutil.copy2(source, destination)

    print(
        class_name,
        "-> Train:",
        len(train_images),
        "Validation:",
        len(val_images)
    )

print("\nDataset splitting completed.")