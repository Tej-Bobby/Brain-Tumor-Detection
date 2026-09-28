import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict/"


st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


st.title(" Brain Tumor Detection")

st.write(
    "Upload an MRI image to classify the tumor type using a trained "
    "MobileNetV2 deep learning model."
)


st.divider()


# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("🔍 Predict Tumor", use_container_width=True):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Analyzing MRI image..."):

            response = requests.post(
                API_URL,
                files=files
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            st.success(
                f"Prediction: {prediction}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Class Probabilities")

            for class_name, probability in probabilities.items():

                st.write(
                    f"{class_name.capitalize()}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

        else:

            st.error(
                "Prediction failed. Please try again."
            )


# Prediction history
st.divider()

st.subheader("📋 Prediction History")


if st.button("View History", use_container_width=True):

    history_response = requests.get(
        "http://127.0.0.1:8000/predict/history"
    )

    if history_response.status_code == 200:

        history = history_response.json()

        if history:

            for record in history:

                st.write(
                    f"**#{record['id']}**  |  "
                    f"{record['prediction']}  |  "
                    f"{record['confidence'] * 100:.2f}%  |  "
                    f"{record['created_at']}"
                )

        else:

            st.info("No prediction history found.")

    else:

        st.error(
            "Could not load prediction history."
        )
