import streamlit as st
from pypdf import PdfReader

# from google.cloud import texttospeech
# import io
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("PDF to Speech Converter")

# Upload a PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file:
    # Read PDF
    pdf_reader = PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"PDF pages: {num_pages}")

    # Let user select a page to convert
    page_num = st.number_input(
        "Select page number", min_value=1, max_value=num_pages, value=1
    )

    # Extract text from chosen page
    page = pdf_reader.pages[page_num - 1]
    text = page.extract_text()

    if not text:
        st.error("No text found on the selected page.")
    else:
        st.write("Page text preview:")
        st.text_area("Text preview", text, height=200, label_visibility="collapsed")

        # Button to generate audio
        if st.button("Generate Audio"):
            # # Initialize Google TTS client
            # client = texttospeech.TextToSpeechClient()

            # synthesis_input = texttospeech.SynthesisInput(text=text)
            # voice = texttospeech.VoiceSelectionParams(
            #     language_code="ar-XA",  # adjust as needed
            #     name="ar-XA-Standard-B",
            #     ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            # )
            # audio_config = texttospeech.AudioConfig(
            #     audio_encoding=texttospeech.AudioEncoding.MP3
            # )

            # response = client.synthesize_speech(
            #     input=synthesis_input, voice=voice, audio_config=audio_config
            # )

            # # Save audio into a buffer
            # audio_buffer = io.BytesIO(response.audio_content)

            # st.success("Audio generated successfully!")
            # st.audio(audio_buffer.getvalue(), format="audio/mp3")
            # st.download_button(
            #     label="Download Audio",
            #     data=response.audio_content,
            #     file_name=f"page_{page_num}.mp3",
            #     mime="audio/mp3",
            # )

            # Initialize OpenAI TTS client
            client = OpenAI()
            speech_file_path = Path(__file__).parent / "speech.mp3"
            response = client.audio.translations.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            response.write_to_file(speech_file_path)

            # Read the generated audio file
            with open(speech_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            st.success("Audio generated successfully!")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name=f"page_{page_num}.mp3",
                mime="audio/mp3",
            )
