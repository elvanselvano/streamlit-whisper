import streamlit as st

st.set_page_config(
    page_title="Starting App",
)

def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Starting app")

if __name__ == "__main__":
    main()