import os
from io import BytesIO

import numpy as np
import whisper
import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from agents.blueprints import financial_planner
from elevenlabs.client import ElevenLabs

st.set_page_config(
    layout="centered",
    page_title="FinNetra",
)


@st.cache_resource
def load_whisper_model():
    """
    Load and cache the medium size OpenAI Whisper model.
    Returns:
        object: The loaded Whisper model.
    """
    model = whisper.load_model("medium")
    return model


def transcribe_audio(model, audio):
    """
    Transcribe audio using the OpenAI Whisper model.
    This function takes an audio object, ensures that the audio parameters
    (frame rate, sample width, and channels) meet OpenAI Whisper specifications,
    converts the audio data into a normalized numpy array, and then transcribes
    it using the Whisper model.
    Args:
        model: The Whisper model used for audio transcription.
        audio: The audio object to be transcribed.
    Returns:
        str: The transcribed audio result in text format.
    """
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)
    if audio.channels != 1:
        audio = audio.set_channels(1)

    audio_array = np.array(audio.get_array_of_samples())
    normalization_factor = 32768.0
    audio_array = audio_array.astype(np.float32) / normalization_factor

    options = dict(
        language=os.environ["WHISPER_MODEL_LANGUAGE"], beam_size=5, best_of=5
    )

    transcribe_options = dict(task="transcribe", **options)
    with st.spinner("Saya sedang berpikir.."):
        result = model.transcribe(audio_array, **transcribe_options)
    return result["text"]


def speak(text: str):
    """
    Convert text into audio and play it.
    Args:
        text: str = the text to be converted into speech
    """
    client = ElevenLabs(
        api_key=os.environ["TTS_API_KEY"]
    )
    audio = client.generate(
        text=text,
        voice=os.getenv("TTS_VOICE", "Rachel"),
        model=os.getenv("TTS_MODEL", "eleven_multilingual_v2")
    )
    audio_file = BytesIO()
    for chunk in audio:
        if chunk:
            audio_file.write(chunk)
    st.audio(audio_file, autoplay=True)


def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("FinNetra")
    st.write("Menyediakan tunanetra hak atas akses finansial yang setara melalui kekuatan AI dan ML.")

    if "chat" not in st.session_state:
        speak("Halo! Kenalin nama, profil, dan kondisi keuangan kamu dong!")
        st.session_state["chat"] = []

    audio = audiorecorder("Mulai merekam", "Berhenti merekam")

    if len(audio) > 0:
        model = load_whisper_model()
        text = transcribe_audio(model, audio)
        st.session_state["chat"].append({"role": "Nasabah", "message": text})
        with st.spinner("Membuat perencanaan keuanganmu.."):
            planner = financial_planner()
            res = planner.run(
                {
                    "chat": {"value": st.session_state["chat"]},
                },
            )
            response = res["financial_planner"]["replies"][0]
        speak(response)
        st.session_state["chat"].append({"role": "Anda", "message": response})
        for i in st.session_state["chat"]:
            st.write(f"{i['role']}: {i['message']}")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
