import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from data_loader import val_dataset


MODEL_PATH = "models/brain_tumor_mobilenet_preprocessed.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

y_true = []
y_pred = []


for images, labels in val_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)