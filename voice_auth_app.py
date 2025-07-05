import streamlit as st
import sounddevice as sd
import numpy as np
import joblib
import os
import tempfile
from scipy.io.wavfile import write
from extract_features import extract_mfcc

# Load trained model
model = joblib.load("voice_model.pkl")

# Streamlit UI
st.set_page_config(page_title="WhisperAuth", page_icon="🔐", layout="centered")
st.title("🔐 WhisperAuth - Voice Based Login")
st.write("Please speak your passphrase clearly into the microphone")

# Record voice function
def record_voice(duration=3, fs=44100):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "voice_input.wav")
    write(file_path, fs, audio)
    return file_path

# Authenticate function
def authenticate(file_path):
    features = extract_mfcc(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return prediction, probs

# Button to record and authenticate
if st.button("🎙️ Start Voice Authentication"):
    file_path = record_voice()
    prediction, probs = authenticate(file_path)

    if prediction == 1:
        st.success(f"✅ Access Granted! Confidence: {probs[1]*100:.2f}%")
        st.balloons()
    else:
        st.error(f"❌ Access Denied. Confidence: {probs[0]*100:.2f}%")
        st.image("https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif", caption="Sorry, not recognized", use_column_width=True)

st.caption("Developed with ❤️ for your M.Tech Project")
