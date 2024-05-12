import streamlit as st
from audiorecorder import audiorecorder

st.set_page_config(
  page_title="Starting App",
)

def main():
  st.markdown("<br>", unsafe_allow_html=True)
  st.title("Starting app")

  st.title("Audio Recorder")
  audio = audiorecorder(start_prompt="", stop_prompt="", pause_prompt="", show_visualizer=True)

  if len(audio) > 0:
      st.audio(audio.export().read())

      audio.export("audio.wav", format="wav")

      st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

def test_main():
  pass

if __name__ == "__main__":
  main()