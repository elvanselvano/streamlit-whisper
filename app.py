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
        language = 'Indonesian'
        options = dict(language=language, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0, **transcribe_options)
        st.write(result['text'])

if __name__ == "__main__":
    main()
