import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = "models/brain_tumor_mobilenet_preprocessed.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(IMG_SIZE)

    # Convert image to numpy array
    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    # Make prediction
    predictions = model.predict(image, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    # Get confidence
    confidence = float(predictions[0][predicted_index])

    # Get probabilities for all classes
    probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probabilities


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predicted_class, confidence, probabilities = predict_image(
        image_path
    )

    print("\nPrediction:", predicted_class)
    print("Confidence:", f"{confidence * 100:.2f}%")

    print("\nAll Class Probabilities:")

    for class_name, probability in probabilities.items():
        print(
            f"{class_name}: {probability * 100:.2f}%"
        )