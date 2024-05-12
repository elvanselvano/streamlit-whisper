import streamlit as st
from audiorecorder import audiorecorder
import whisper
import numpy as np

st.set_page_config(
    page_title="Starting App",
)

def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Starting app")
    audio = audiorecorder("Click to record", "Click to stop recording")

    if len(audio) > 0:
        model = whisper.load_model("base")
        result = model.transcribe(np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0)
        st.write(result['text'])

if __name__ == "__main__":
    main()