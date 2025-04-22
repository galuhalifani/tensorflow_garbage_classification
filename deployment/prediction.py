import streamlit as st
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

with open("model_config.json", "r") as f:
    config = json.load(f)

BATCH_SIZE = config["BATCH_SIZE"]
img_height = config["img_height"]
img_width = config["img_width"]
IMG_SIZE = tuple(config["IMG_SIZE"])
class_names = config["class_names"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

def load_inference_image(file):
    img = tf.keras.utils.load_img(file, target_size=(img_height,img_width))
    x = tf.keras.utils.img_to_array(img)
    plt.imshow(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    return images

def print_category(item):
    with open("categorization.json", "r") as c:
        categorization = json.load(c)
    
    for category, items in categorization.items():
        if item in items:
            return f"**Category**: {category}"
    return f"**{item}** not found in any category."

def prediction(img_file):
  model = load_model()
  images = load_inference_image(img_file)
  y_pred_proba = model.predict(images)
  y_pred_class = np.argmax(y_pred_proba[0])
  y_pred_class_name = class_names[y_pred_class]
  return y_pred_class_name

def run():
    st.markdown("## Garbage Image Classifier")
    st.markdown("---")

    col1, col2, col3 = st.columns([2.5, 0.2, 1.5])

    with col1:
        uploaded_file = st.file_uploader(
            "### Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Only JPG, JPEG, PNG formats supported"
        )

        image_slot = st.empty()

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_slot.image(image, caption=uploaded_file.name, use_container_width=True)
        else:
            image_slot.empty()

    with col3:
        st.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: green !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
        submitted = st.button('🔍 Classify Image', use_container_width=True)

        if submitted:
            status_placeholder = st.empty()
            result_slot_1 = st.empty()
            result_slot_2 = st.empty()

            if uploaded_file is not None:
                status_placeholder.markdown("⏳ **Running classification...**")
                predicted_class = prediction(uploaded_file)
                treatment_info = print_category(predicted_class)
                status_placeholder.empty()
                result_slot_1.success(f"**Predicted Class:** {predicted_class}")
                result_slot_2.info(f"{treatment_info}")
            else:
                result_slot_1.empty()
                result_slot_2.empty()
                st.warning("No image uploaded")

if __name__ == '__main__':
    run()