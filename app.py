import streamlit as st
import numpy as np
from PIL import Image, ExifTags
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os
import gdown
import pandas as pd
import folium
from streamlit_folium import st_folium

MODEL_FILE = "mapillary_model_fast.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=141uXNq3hfhTBlxzGBZBzVMLrFs8cA5XX"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("📥 Κατεβάζουμε το μοντέλο..."):
            gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

def get_exif_gps(img):
    try:
        exif_data = img._getexif()
        if not exif_data:
            return None
        for key, val in ExifTags.TAGS.items():
            if val == 'GPSInfo':
                gps_info = exif_data.get(key)
                if not gps_info:
                    return None
                def to_degrees(value):
                    d, m, s = value
                    return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600
                lat = to_degrees(gps_info[2])
                if gps_info[1] == 'S': lat *= -1
                lon = to_degrees(gps_info[4])
                if gps_info[3] == 'W': lon *= -1
                return lat, lon
    except Exception:
        return None
    return None

model = load_cnn_model()
IMG_SIZE = (128, 128)

st.title("📍 Mapillary Road Sign Classifier + GPS Map")
st.write("Ανεβάστε εικόνες πινακίδων για ταξινόμηση: `damaged` ή `not_damaged`, με δυνατότητα γεωτοποθέτησης")

threshold = st.slider("📊 Threshold ταξινόμησης (damaged):", 0.0, 1.0, 0.8, 0.01)
uploaded_files = st.file_uploader("Επιλέξτε εικόνες...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = []

gps_points = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prob = model.predict(img_array)[0][0]
        label = "damaged" if prob >= threshold else "not_damaged"

        st.image(img, caption=f"{uploaded_file.name}", use_column_width=True)
        st.markdown(f"**Prediction:** `{label}`")
        st.write(f"Probability: {round(prob, 3)}")

        latlon = get_exif_gps(img)
        if latlon:
            gps_points.append({
                "filename": uploaded_file.name,
                "label": label,
                "probability": round(prob, 3),
                "location": latlon
            })

        st.session_state.history.append({
            "Filename": uploaded_file.name,
            "Label": label,
            "Probability": round(prob, 3),
            "Threshold": round(threshold, 2)
        })

if st.session_state.history:
    st.subheader("📜 Ιστορικό Προβλέψεων")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

if gps_points:
    st.subheader("🗺️ Χάρτης Πινακίδων με GPS")
    fmap = folium.Map(location=gps_points[0]["location"], zoom_start=12)
    for p in gps_points:
        folium.Marker(
            location=p["location"],
            popup=f"{p['filename']} ({p['label']}, p={p['probability']})",
            icon=folium.Icon(color="red" if p["label"] == "damaged" else "green")
        ).add_to(fmap)
    st_folium(fmap, width=700, height=500)
