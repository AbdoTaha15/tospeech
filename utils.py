import os
from pypdf import PdfReader
import base64
import io
import streamlit as st
import openai
from pydub import AudioSegment

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_image_pdf(img_data: bytes):
    """Extract text from an image-based PDF page using OpenAI Vision."""
    try:
        base64_image = base64.b64encode(img_data).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the extracted text, no explanations or other text.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using PyPDF."""
    reader = PdfReader(pdf_path)
    results = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Check if text is not empty
            results.append({"page": i + 1, "text": text, "type": "text"})
        else:
            images = page.images
            if images:
                for img in images:
                    # If the page contains images, we assume it's an image-based PDF
                    results.append(
                        {
                            "page": i + 1,
                            "text": extract_text_from_image_pdf(img.data),
                            "type": "image",
                        }
                    )
            else:
                # If the page is blank, we can skip it or handle it as needed
                continue
    return results


def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI API."""
    try:
        # Available voices: alloy, echo, fable, onyx, nova, shimmer
        completion = client.chat.completions.create(
            model="gpt-4o-mini-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "wav"},
            messages=[
                {
                    "role": "developer",
                    "content": "Convert the exact input text to speech without adding any additional content, commentary, or explanations. Detect the language and use authentic pronunciation, intonation, and speaking style for that language. The audio should contain only the user's text verbatim.",
                },
                {"role": "user", "content": text},
            ],
        )
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        # Convert the binary response content to a BytesIO object
        audio_data = io.BytesIO(wav_bytes)
        audio_data.seek(0)
        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None


def merge_audio_files(audio_files_dict, pages_data):
    """
    Merge multiple audio files into a single audio file.

    Parameters:
    - audio_files_dict: Dictionary with page indices as keys and audio data as values
    - pages_data: List of page data objects

    Returns:
    - BytesIO object containing merged audio data in MP3 format
    """
    try:
        # Create a silent audio segment to start with
        merged = AudioSegment.silent(duration=500)  # Start with 500ms silence

        # Sort the keys to process pages in order
        sorted_keys = sorted(audio_files_dict.keys())

        for idx in sorted_keys:
            # Get the audio data for this page
            audio_data = audio_files_dict[idx]
            audio_data.seek(0)

            # Load the audio segment from the BytesIO object
            segment = AudioSegment.from_file(audio_data, format="wav")

            # Add a short pause between pages (1 second)
            pause = AudioSegment.silent(duration=1000)

            # Add the current segment to the merged audio
            merged = merged + segment + pause

        # Export the merged audio to a BytesIO object in MP3 format
        output = io.BytesIO()
        merged.export(output, format="mp3")
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error merging audio files: {str(e)}")
        return None
