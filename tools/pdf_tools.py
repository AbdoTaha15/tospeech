import os
import base64
import io
import tempfile
import streamlit as st
import openai
from google import genai
from google.genai import types
from pydub import AudioSegment
from pdf2image import convert_from_path
from PIL import Image  # Add PIL import for image processing

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))


def extract_text_from_image_openai(image):
    """Extract text from an image-based PDF page using OpenAI Vision."""
    try:
        # Save the PIL Image to a BytesIO buffer as PNG
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        # Get base64 encoded string
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": "Your task is to perform optical character recognition on an image containing Arabic text. You are a specialized Arabic OCR agent with expertise in Arabic script. Your output must be an exact and complete transcription of the text as it appears in the image. This means capturing every character, diacritic, punctuation mark, spacing, and formatting detail with perfect accuracy. Do not omit, alter, or add any content. Your response should consist solely of the plain Arabic text, faithfully mirroring the original image.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")


def extract_text_from_image_gemini(image):
    """Extract text from an image-based PDF page using Google Gemini."""
    try:
        # Convert PIL image to an IOBase object for uploading
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)  # Reset file pointer to beginning

        # Upload the image to Gemini
        file_ref = gemini_client.files.upload(
            file=buffered,
            config=types.UploadFileConfig(
                mime_type="image/png",
            ),
        )

        # Use Gemini model to extract text
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="""You are an advanced Optical Character Recognition (OCR) engine specialized in processing images containing Arabic text. Your task is to extract all Arabic text from the provided image file with the highest accuracy. Please adhere to the following instructions:

1. **Input Processing:**  
   - Process the attached image file, which contains Arabic text.

2. **Text Extraction:**  
   - Identify and extract every instance of Arabic text from the image.  
   - Maintain the original order of the text as it appears in the image.  
   - Preserve all punctuation, diacritics, and formatting present in the original text.

3. **Output Requirements:**  
   - Return the extracted text in plain text format without any additional commentary or interpretation.  
   - If any part of the image is unclear or ambiguous, mark that segment with a placeholder (e.g., "[?]") to indicate uncertainty.

4. **Error Handling:**  
   - If the image contains non-textual elements (graphics, background noise, etc.), ignore them and focus solely on the Arabic text content.

Begin processing the image now and output only the text content you extract.
""",
                temperature=0.1,
            ),
            contents=[
                file_ref,
            ],
        )

        # Clean up
        gemini_client.files.delete(name=file_ref.name)

        return response.text
    except Exception as e:
        raise Exception(f"Error extracting text with Gemini: {str(e)}")


def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using PyPDF."""
    # Convert PDF to a list of PIL Image objects
    pages = convert_from_path(pdf_path, dpi=500, fmt="png", thread_count=4)
    results = []

    for i, page in enumerate(pages, start=1):
        results.append(
            {
                "page": i,
                "text": extract_text_from_image_openai(page),
            }
        )
        print(f"Extracted text from page {i}")
    return results


def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI API."""
    try:
        audio_path = None
        with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=text,
        ) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                response.stream_to_file(temp_audio.name)
                audio_path = temp_audio.name

        # Read the audio file directly into a BytesIO object
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # # Available voices: alloy, echo, fable, onyx, nova, shimmer
        # completion = openai_client.chat.completions.create(
        #     model="gpt-4o-audio-preview",
        #     modalities=["text", "audio"],
        #     audio={"voice": voice, "format": "wav"},
        #     messages=[
        #         {
        #             "role": "developer",
        #             "content": "Please convert the following text into speech using a native, expert-level Arabic accent. Ensure that every word is pronounced accurately and completely, strictly following the exact formatting, punctuation, and spacing of the text. Do not skip, modify, or add any words or extra commentary. Your output should be the precise spoken rendition of the text provided.",
        #         },
        #         {"role": "user", "content": text},
        #     ],
        # )

        # Convert the binary data to a BytesIO object
        audio_data = io.BytesIO(audio_bytes)
        audio_data.seek(0)

        # Clean up the temporary file
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None


def merge_audio_files(audio_files_dict):
    """
    Merge multiple audio files into a single audio file.

    Parameters:
    - audio_files_dict: Dictionary with page indices as keys and audio data as values

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
