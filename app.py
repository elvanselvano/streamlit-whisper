import streamlit as st
from audiorecorder import audiorecorder
import whisper
import numpy as np
from pydub import AudioSegment
import librosa

st.set_page_config(
    page_title="Starting App",
)

@st.cache
def load_whisper_model():
    model = whisper.load_model("medium")
    return model

def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Starting app")

    model = load_whisper_model()
    st.success('Whisper model loaded.')

    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        audio.export("audio.mp3", format="mp3")
        st.success("Recording saved. Now transcribing audio...")

        y, _ = librosa.load('audio.mp3')
        language = 'Indonesian'
        options = dict(language=language, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(y, **transcribe_options)

        st.write('Transcribed audio:')
        st.write(result['text'])

if __name__ == "__main__":
    main()
