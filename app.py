import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.explainability import make_gradcam, overlay_heatmap

MODEL_PATH = Path("models/plant_disease_mobilenetv2.keras")
CLASSES_PATH = Path("models/class_names.json")

st.set_page_config(page_title="Plant Disease Detection + XAI", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Detection using Deep Learning & Explainable AI")
st.write("Upload a plant-leaf image to classify it and inspect a Grad-CAM explanation.")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return model, classes

model, class_names = load_artifacts()
if model is None:
    st.warning("No trained model found. Follow the README instructions to prepare the dataset and run `python train.py`.")
    st.stop()

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    resized = image.resize((224, 224))
    batch = np.expand_dims(np.array(resized).astype("float32"), axis=0)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    with st.spinner("Analyzing leaf image..."):
        probs = model.predict(preprocessed, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        heatmap = make_gradcam(model, preprocessed, class_index=pred_idx)
        overlay = overlay_heatmap(np.array(resized), heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(image_np, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM explanation")
        st.image(overlay, caption="Highlighted regions influenced the model's prediction", use_container_width=True)

    st.subheader("Prediction")
    st.success(f"Predicted class: **{class_names[pred_idx]}**")
    st.metric("Model confidence score", f"{probs[pred_idx]*100:.2f}%")
    st.caption("Confidence is a model score, not a guarantee of correctness. Grad-CAM shows model attention, not biological proof.")

    st.subheader("Class scores")
    order = np.argsort(probs)[::-1][:min(5, len(class_names))]
    for idx in order:
        st.write(f"{class_names[idx]} — {probs[idx]*100:.2f}%")
        st.progress(float(probs[idx]))

    with st.expander("Optional LIME explanation"):
        st.write("LIME is optional and may take some time. Install `lime` and `scikit-image` to enable it.")
        if st.checkbox("Generate LIME explanation"):
            try:
                from lime import lime_image
                from skimage.segmentation import mark_boundaries
                explainer = lime_image.LimeImageExplainer()
                def predict_fn(images):
                    arr = np.array(images).astype("float32")
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                    return model.predict(arr, verbose=0)
                explanation = explainer.explain_instance(
                    np.array(resized), predict_fn, top_labels=1,
                    hide_color=0, num_samples=500
                )
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], positive_only=True, num_features=8, hide_rest=False
                )
                st.image(mark_boundaries(temp / 255.0, mask),
                         caption="LIME: superpixels supporting the selected class",
                         use_container_width=True)
            except Exception as exc:
                st.error(f"Could not create LIME explanation: {exc}")

st.divider()
st.caption("Educational prototype. Confirm suspected diseases with a qualified agricultural extension officer or plant pathologist.")
