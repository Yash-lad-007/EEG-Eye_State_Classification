import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource 
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="EEG Eye State Prediction", layout="centered")

st.title("EEG Eye State Classification")
st.write("Predict whether **Eyes are Open or Closed** using EEG signals.")

st.markdown("---")

# ---------------------------
# Input fields (14 EEG features)
# ---------------------------
st.subheader("Enter EEG Channel Values")

features = []
feature_names = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

for feature in feature_names:
    value = st.number_input(
        f"{feature}",
        value=0.0,
        format="%.5f"
    )
    features.append(value)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Eye State"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Eyes Closed")
    else:
        st.info("Eyes Open")
