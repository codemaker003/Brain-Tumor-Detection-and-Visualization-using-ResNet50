import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ✅ Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C://Users//donka//Downloads//modelres50.h5")

model = load_model()
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ✅ Preprocessing
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (200, 200))
    normalized = resized / 255.0
    stacked = np.stack((normalized,) * 3, axis=-1)  # Convert to 3 channels
    return np.expand_dims(stacked, axis=0)

# ✅ Tumor + Brain Region Highlighting
def highlight_tumor_regions(image, predicted_class):
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for brain region (exclude black background)
    _, brain_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Blur and threshold for tumor detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, tumor_mask = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)

    overlay = image.copy()

    # Apply light green for brain
    overlay[brain_mask == 255] = [144, 238, 144]  # Light green

    # Apply light red for tumor
    if predicted_class != 2:  # Not 'no_tumor'
        overlay[tumor_mask == 255] = [255, 105, 105]  # Light red

    # Combine overlay only inside brain
    final_image = image.copy()
    final_image[brain_mask == 255] = cv2.addWeighted(
        image, 0.6, overlay, 0.4, 0
    )[brain_mask == 255]

    return final_image

# ✅ Streamlit UI
st.title("🧠 Brain Tumor Detection with Tumor Highlighting")
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original MRI Scan")
    st.image(img_array, use_column_width=True)

    # Predict
    input_data = preprocess_image(img_array)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    st.subheader("Prediction Results")
    st.markdown(f"**Class:** `{classes[predicted_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Highlight
    st.subheader("Tumor Highlighted Output")
    highlighted_img = highlight_tumor_regions(img_array, predicted_class)
    st.image(highlighted_img, use_column_width=True)
