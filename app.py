import streamlit as st
from dotenv import load_dotenv
from agents.blueprints import profile_extractor
from audiorecorder import audiorecorder

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

    # audio = audiorecorder("Click to record", "Click to stop recording")
    placeholder_story = result['text']

    # if len(audio) > 0:
    # model = whisper.load_model("base")
    # language = 'Indonesian'
    # options = dict(language=language, beam_size=5, best_of=5)
    # transcribe_options = dict(task="transcribe", **options)
    # result = model.transcribe(np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0, **transcribe_options)
    profile_extractor_pipe = profile_extractor()
    with st.spinner("Wait for it..."):
        res = profile_extractor_pipe.run({"prompt": {"value": placeholder_story}})
        reply = res["generator"]["replies"][0]
    st.write(reply)
    # print(profile_extractor_pipe.run({'prompt':{"value": placeholder_story}}))

if __name__ == "__main__":
    load_dotenv(".env")
    main()
