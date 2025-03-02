import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


pg = st.navigation(
    [
        st.Page("views/pdf_to_audio.py", title="PDF to Audio", icon="📄"),
        st.Page("views/video_translation.py", title="Video Translation", icon="🎥"),
    ]
)
pg.run()
