import tensorflow as tf

from data_loader import train_dataset, val_dataset
from transforms import model

EPOCHS = 10

model.layers[0].trainable = False

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

model.save("models/brain_tumor_mobilenet_preprocessed.keras")

print("\nMobileNetV2 training completed.")
print("Model saved to: models/brain_tumor_mobilenet_preprocessed.keras")