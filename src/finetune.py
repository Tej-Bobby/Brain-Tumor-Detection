import tensorflow as tf

from data_loader import train_dataset, val_dataset


MODEL_PATH = "models/brain_tumor_mobilenet.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# Find the MobileNetV2 base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break


# Unfreeze the base model
base_model.trainable = True


# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# Recompile with a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# Fine-tune
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)


# Save fine-tuned model
model.save("models/brain_tumor_mobilenet_finetuned.keras")

print("\nFine-tuning completed.")
print("Model saved to: models/brain_tumor_mobilenet_finetuned.keras")