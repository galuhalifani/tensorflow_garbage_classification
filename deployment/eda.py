import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from glob import glob
import cv2
import numpy as np

base_dir = "./garbage_classification"

plt.rcParams.update({
    "axes.labelpad": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def get_image_info(path):
    with Image.open(path) as img:
        width, height = img.size
    return pd.Series([width, height, width / height])

def compute_image_info(df):
    df[['width', 'height', 'aspect_ratio']] = df['images'].apply(get_image_info)
    return df

def create_chart(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    category_counts = df['category'].value_counts()
    ax.bar(category_counts.index, category_counts.values)
    ax.set_title(f"Distribution of classes")
    ax.set_xlabel('class')
    ax.set_ylabel("Count")
    st.pyplot(fig)

def detect_blur(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    return cv2.Laplacian(img, cv2.CV_64F).var()

def create_dataframe(image_path):
    cardboard_files = glob(f'{image_path}/cardboard/*')
    paper_files = glob(f'{image_path}/paper/*')
    plastic_files = glob(f'{image_path}/plastic/*')
    metal_files = glob(f'{image_path}/metal/*')
    glass_files = glob(f'{image_path}/glass/*')
    trash_files = glob(f'{image_path}/trash/*')

    all_files = (
        cardboard_files +
        paper_files +
        plastic_files +
        metal_files +
        glass_files +
        trash_files
    )

    cardboard = ['cardboard' for i in range(len(cardboard_files))]
    paper = ['paper' for i in range(len(paper_files))]
    plastic = ['plastic' for i in range(len(plastic_files))]
    metal = ['metal' for i in range(len(metal_files))]
    glass = ['glass' for i in range(len(glass_files))]
    trash = ['trash' for i in range(len(trash_files))]

    category = cardboard + paper + plastic + metal + glass + trash

    img_df = pd.DataFrame({
        'images': all_files,
        'category': category
    })

    df = compute_image_info(img_df)
    df['sharpness'] = df['images'].apply(detect_blur)
    df['images'] = [os.path.basename(f) for f in all_files]
    return df

def count_blurred(blur_threshold, img_df):
    blurry_images = img_df[img_df['sharpness'] < blur_threshold]
    total_per_class = img_df['category'].value_counts().sort_index()
    blurry_per_class = blurry_images['category'].value_counts().reindex(total_per_class.index, fill_value=0)
    blur_percentage = (blurry_per_class / total_per_class) * 100

    st.subheader('Blurry Image Analysis')
    blurry_per_class_sorted = blurry_per_class.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(blurry_per_class_sorted.index, blurry_per_class_sorted.values)
    ax.set_title(f"Blurry Image Count per Class")
    ax.set_xlabel('class')
    ax.set_ylabel("Count")
    st.pyplot(fig)

    print('\n')
    blur_percentage_sorted = blur_percentage.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(blur_percentage_sorted.index, blur_percentage_sorted.values)
    ax.set_title(f"Blurry Image Percentage per Class")
    ax.set_xlabel('class')
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)

def run():
    html_style = '''
        <style>
        .block-container {
            padding-left: 5rem;
            padding-right: 15rem;
            margin-left: 15rem !important;
            margin-right: 15rem !important;
            margin-top: -3rem!important;
        }
        </style>
    '''
    st.markdown(html_style, unsafe_allow_html=True)
    st.title('Garbage Recycling Classification')
    st.image("recycling.jpg", use_container_width=True)
    with st.expander("Dataset Information", expanded=False):
        st.markdown(
        "<p style='font-size:12px;'>"
            "Dataset is retrived from Stanford research paper by Yang, Mindy and Thung, Gary, which was also published in Kaggle. The Garbage Classification Dataset contains 2.5k images from 6 garbage categories, which were taken by placing the object on a white posterboard and using sunlight and/or room lighting."
            "<br><br> See sidebar for data source reference and credits."
        "</p>", 
        unsafe_allow_html=True)
    st.markdown('---')
    
    df = create_dataframe(base_dir)
    st.subheader('Class Distribution')
    create_chart(df)

    st.markdown('---')
    st.subheader('Image Samples')
    className = st.selectbox('Select a class:', ['plastic', 'glass', 'paper', 'metal', 'cardboard', 'trash'])
    num_images = st.selectbox("How many images to display?", [9, 15, 30, 60], index=0)
    image_dir = os.path.join(base_dir, className)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        st.warning(f"No images found in `{className}` folder.")
        return

    cols = st.columns(3)
    for idx, file in enumerate(image_files[:num_images]):
        img_path = os.path.join(image_dir, file)
        image = Image.open(img_path)
        with cols[idx % 3]:
            st.image(image, caption=file, use_container_width=True)

    count_blurred(30, df)
    st.markdown('---')
    st.header('Raw Sample Summary')
    st.dataframe(df)

if __name__ == '__main__':
    run()