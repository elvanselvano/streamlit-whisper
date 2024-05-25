import os
from io import BytesIO

import numpy as np
import whisper
import streamlit as st
from gtts import gTTS
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from agents.blueprints import profile_extractor

st.set_page_config(
    page_title="Starting App",
)


@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("medium")
    return model


def transcribe_audio(model, audio):
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)
    if audio.channels != 1:
        audio = audio.set_channels(1)

    audio_array = np.array(audio.get_array_of_samples())
    normalization_factor = 32768.0
    audio_array = audio_array.astype(np.float32) / normalization_factor
    st.success("Transcribing audio...")

    options = dict(
        language=os.environ["WHISPER_MODEL_LANGUAGE"], beam_size=5, best_of=5
    )

    transcribe_options = dict(task="transcribe", **options)
    result = model.transcribe(audio_array, **transcribe_options)
    return result["text"]


def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Starting app")

    model = load_whisper_model()
    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        transcribed_audio_text = transcribe_audio(model, audio)
        st.write("Transcribed audio:")
        st.write(transcribed_audio_text)

        profile_extractor_pipe = profile_extractor()
        with st.spinner("Wait for it..."):
            res = profile_extractor_pipe.run(
                {"prompt": {"value": transcribed_audio_text}}
            )
            reply = res["generator"]["replies"][0]
        st.write(reply)

        lang = os.environ["GTTS_MODEL_LANGUAGE"]
        tts = gTTS(text=transcribed_audio_text, lang=lang)
        audio_bytes = BytesIO()

        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes, format="audio/mp3")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
