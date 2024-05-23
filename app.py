import streamlit as st
from audiorecorder import audiorecorder
import whisper
import numpy as np

st.set_page_config(
    page_title="Starting App",
)

@st.cache_resource
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
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        if audio.channels != 1:
            audio = audio.set_channels(1)
        audio_array = np.array(audio.get_array_of_samples())
        audio_array = audio_array.astype(np.float32) / 32768.0
        st.success("Recording saved. Now transcribing audio...")

        language = 'Indonesian'
        options = dict(language=language, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(audio_array, **transcribe_options)

        st.write('Transcribed audio:')
        st.write(result['text'])

if __name__ == "__main__":
    main()
