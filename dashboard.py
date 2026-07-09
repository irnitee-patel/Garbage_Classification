import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from PIL import Image
# Load trained model
model = load_model("model/garbage_classifier.keras")
class_names = [
    "Battery",
    "Cardboard",
    "Glass",
    "Paper",
    "Plastic"
]
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Garbage Classification using CNN")

st.write(
    "Upload a garbage image and the CNN model will classify it into one of the five categories."
)
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
##precessed image
    image = image.resize((128, 128))

    img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)
##predict the image
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    predicted_class = class_names[predicted_index]
##display prediction
    st.success(f"Predicted Class : {predicted_class}")

    st.info(f"Confidence : {confidence*100:.2f}%")
##show confidence for every class
    st.subheader("Prediction Scores")

    probability = prediction[0]

    for label, score in zip(class_names, probability):

        st.write(f"{label} : {score*100:.2f}%")

##display bar chart

    st.subheader("Prediction Probability Chart")

    chart_data = {
        "Class": class_names,
        "Probability": probability
    }

    st.bar_chart(
        data=chart_data,
        x="Class",
        y="Probability"
    )  
