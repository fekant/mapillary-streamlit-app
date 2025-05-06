import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os
import gdown

MODEL_FILE = "mapillary_model_fast.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=141uXNq3hfhTBlxzGBZBzVMLrFs8cA5XX"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("📥 Κατεβάζουμε το μοντέλο..."):
            gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

model = load_cnn_model()
THRESHOLD = 0.8
IMG_SIZE = (128, 128)

st.title("🚧 Mapillary Road Sign Classifier")
st.write("Ανεβάστε εικόνα πινακίδας για ταξινόμηση: `damaged` ή `not_damaged`")

uploaded_file = st.file_uploader("Επιλέξτε εικόνα...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Επιλεγμένη εικόνα", use_column_width=True)

    img_resized = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "damaged" if prob >= THRESHOLD else "not_damaged"

    st.markdown(f"### 🔎 Prediction: `{label}`")
    st.write("Probability: {:.3f}".format(prob))
