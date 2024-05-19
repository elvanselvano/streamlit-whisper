import streamlit as st
from audiorecorder import audiorecorder
from dotenv import load_dotenv
import whisper
import numpy as np
from agents.blueprints import profile_extractor

st.set_page_config(
    page_title="Starting App",
)

def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Starting app")
    # audio = audiorecorder("Click to record", "Click to stop recording")
    placeholder_story = '''
    Nama saya Budi Santoso, seorang pria berusia 35 tahun yang bekerja sebagai Manajer Pemasaran. Saya menikah dan memiliki seorang anak berusia 4 tahun. Setiap bulan, saya menerima gaji sebesar Rp 15.000.000 dan mendapatkan tambahan pendapatan Rp 2.000.000 dari bisnis sampingan. Saat ini, saya memiliki tabungan sebesar Rp 50.000.000 dan telah berinvestasi Rp 20.000.000 di reksa dana.

Namun, saya juga memiliki kewajiban finansial yang harus dipenuhi. Saya sedang mencicil Kredit Pemilikan Rumah (KPR) dengan sisa hutang sebesar Rp 500.000.000, di mana saya harus membayar cicilan bulanan Rp 5.000.000 selama 20 tahun ke depan. Selain itu, saya memiliki hutang kartu kredit sebesar Rp 10.000.000 dengan bunga 2% per bulan.

Pengeluaran bulanan saya terdiri dari cicilan KPR Rp 5.000.000, cicilan kartu kredit Rp 500.000, biaya hidup sehari-hari seperti kebutuhan rumah tangga, sekolah anak, dan transportasi sebesar Rp 8.000.000, serta pengeluaran lain-lain sebesar Rp 2.000.000. Saya juga menyisihkan Rp 1.000.000 setiap bulan untuk tabungan dan investasi. Total pengeluaran bulanan saya mencapai Rp 16.500.000, sehingga saya memiliki sisa pendapatan bulanan Rp 500.000
    '''

    # if len(audio) > 0:
        # model = whisper.load_model("base")
        # language = 'Indonesian'
        # options = dict(language=language, beam_size=5, best_of=5)
        # transcribe_options = dict(task="transcribe", **options)
        # result = model.transcribe(np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0, **transcribe_options)
    profile_extractor_pipe = profile_extractor()
    with st.spinner('Wait for it...'):
        res = profile_extractor_pipe.run({'prompt':{"value": placeholder_story}})
        reply = res['generator']["replies"][0]
    st.write(reply)
    # print(profile_extractor_pipe.run({'prompt':{"value": placeholder_story}}))

if __name__ == "__main__":
    load_dotenv('.env')
    main()
