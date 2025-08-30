import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import io
import os

# -----------------------------
# Load CIFAR-10 Classifier
# -----------------------------
try:
    cifar_classifier = tf.keras.models.load_model("image_classifier.h5")
    cifar_class_names = ["airplane", "automobile", "bird", "cat", "deer",
                         "dog", "frog", "horse", "ship", "truck"]
except Exception as e:
    st.error("❌ Could not load CIFAR-10 model. Please make sure 'image_classifier.h5' exists.")
    cifar_classifier = None
    cifar_class_names = []

# -----------------------------
# Load YOLOv8 Model for COCO
# -----------------------------
yolo_model = YOLO("yolov8n.pt")  # small version, fast; can use yolov8s.pt or yolov8m.pt for more accuracy

# -----------------------------
# Streamlit UI Layout
# -----------------------------
st.set_page_config(page_title="Image Compression & Analysis", layout="wide")

col1, col2 = st.columns([1,5])
with col1:
    st.image("logo.png", width=180)  # place a logo.png in the same folder
with col2:
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Compress and Analyse Images</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size:16px; color: white;'>"
        "Welcome!<br>This page helps you to compress and analyse your image that you upload. Try it out!"
        "</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

st.sidebar.header("Options")
task = st.sidebar.radio("Choose Task:", ["CIFAR-10 Classification", "COCO Detection"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# If an image is uploaded
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # CIFAR-10 Classification
    # -----------------------------
    if task == "CIFAR-10 Classification":
        if cifar_classifier is not None:
            img_resized = image.resize((32, 32))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = cifar_classifier.predict(img_array)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100

            st.markdown(f"<h3 style='color:green;'> Prediction: {cifar_class_names[predicted_class]}</h3>", unsafe_allow_html=True)
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("CIFAR model not loaded. Please check 'image_classifier.h5'.")

    # -----------------------------
    # COCO Detection (YOLOv8)
    # -----------------------------
    elif task == "COCO Detection":
        st.subheader("Object Detection Results")
        results = yolo_model(image)  # run YOLOv8 inference
        detected_objects = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                detected_objects.append(class_name)
        unique_objects = list(set(detected_objects))
        if unique_objects:
            st.success("Objects detected: " + ", ".join(unique_objects))
        else:
            st.warning("No objects detected with high confidence.")
        # Show image with bounding boxes
        st.image(results[0].plot(), caption="Detection Result", use_container_width=True)

    # -----------------------------
    # Compression Option
    # -----------------------------
    st.subheader("Compressed Image")
    compressed_img = image.resize((256, 256), Image.LANCZOS)  # Compress to smaller size
    st.image(compressed_img, caption="Compressed Image (256x256)", use_container_width=True)
    # Download option
    buf = io.BytesIO()
    # Determine whether to resize
    width, height = image.size
    if width > 256 or height > 256:
        compressed_img = image.resize((256, 256), Image.LANCZOS)
    else:
        compressed_img = image.copy()  # keep original size if already small
    # Choose format based on original size / simplicity
        if width <= 256 or height <= 256:
            # Small/simple image → keep PNG
            compressed_img.save(buf, format="PNG", optimize=True)
            file_name = "compressed_image.png"
            mime_type = "image/png"
        else:
            # Larger image → use JPEG for smaller file size
            compressed_img.save(buf, format="JPEG", quality=50)
            file_name = "compressed_image.jpg"
            mime_type = "image/jpeg"
            buf.seek(0)  # move pointer to start of buffer
            # Step 4: Download button
    st.download_button(
        label="Download Compressed Image",
        data=buf,
        file_name=file_name,
        mime=mime_type
    )
    

