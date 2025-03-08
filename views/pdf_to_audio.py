import os
import tempfile
import gc  # Garbage collection
import io
import base64
import streamlit as st
from tools.pdf_tools import (
    extract_text_from_pdf,
    text_to_speech,
    merge_audio_files,
    pdf_to_images,
    extract_text_from_image_openai,
)

if (
    "pdf_page_initialized" not in st.session_state
    or not st.session_state.pdf_page_initialized
):
    # Initialize session state variables
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = {}
    if "batch_progress" not in st.session_state:
        st.session_state.batch_progress = 0
    if "batch_total" not in st.session_state:
        st.session_state.batch_total = 0
    if "batch_processing" not in st.session_state:
        st.session_state.batch_processing = False
    if "batch_page_index" not in st.session_state:
        st.session_state.batch_page_index = 0
    if "pages_data" not in st.session_state:
        st.session_state.pages_data = []
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "merged_audio" not in st.session_state:
        st.session_state.merged_audio = None
    st.session_state.pdf_page_initialized = True

    # Clean up when the app is done
    def cleanup():
        # Safely check if pdf_path exists in session_state before trying to access it
        if (
            "pdf_path" in st.session_state
            and st.session_state.pdf_path
            and os.path.exists(st.session_state.pdf_path)
        ):
            try:
                os.unlink(st.session_state.pdf_path)
            except:
                pass

        # Clean up audio files in session state
        if "audio_files" in st.session_state:
            for file_path in st.session_state.audio_files.values():
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass

    # Register cleanup function to run when the app exits
    import atexit

    atexit.register(cleanup)


# Helper function to convert image to base64 for display
def get_image_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Process the uploaded file only if it's new or different from the previous one
    file_changed = False

    if st.session_state.pdf_path is None:
        file_changed = True
    else:
        # Check if the file has changed
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            new_pdf_path = temp_file.name
            uploaded_file.seek(0)  # Reset file pointer

            if os.path.getsize(new_pdf_path) != os.path.getsize(
                st.session_state.pdf_path
            ):
                file_changed = True
                # Clean up old files
                if os.path.exists(st.session_state.pdf_path):
                    os.unlink(st.session_state.pdf_path)
                st.session_state.pdf_path = new_pdf_path
            else:
                os.unlink(new_pdf_path)

    # If the file is new or has changed, reset session state
    if file_changed:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.pdf_path = temp_file.name

        # Process the PDF
        with st.spinner("Processing PDF..."):
            st.session_state.pages_data = pdf_to_images(st.session_state.pdf_path)
            # Reset other session state variables
            # Clean up old audio files first
            for file_path in st.session_state.audio_files.values():
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass
            st.session_state.audio_files = {}
            st.session_state.batch_progress = 0
            st.session_state.batch_total = 0
            st.session_state.batch_processing = False
            st.session_state.batch_page_index = 0
            st.session_state.merged_audio = None

    # Voice selection for TTS
    voice_options = {
        "Alloy (Neutral)": "alloy",
        "Echo (Balanced)": "echo",
        "Fable (British English)": "fable",
        "Onyx (Deep and authoritative)": "onyx",
        "Nova (Friendly and warm)": "nova",
        "Shimmer (Clear and concise)": "shimmer",
    }
    selected_voice = st.selectbox(
        "Select voice for speech synthesis:", list(voice_options.keys())
    )
    voice = voice_options[selected_voice]

    # Add advanced options expander
    with st.expander("Advanced Options"):
        audio_models_options = {
            "Standard TTS model": "tts-1",
            "High-definition TTS model": "tts-1-hd",
        }

        st.subheader("Audio Quality")
        selected_audio_model = st.selectbox(
            "Select audio quality for speech:",
            list(audio_models_options.keys()),
        )
        audio_model = audio_models_options[selected_audio_model]
        st.markdown("Note: Higher quality models will cost more.")

    # Create a dropdown to select the page
    page_numbers = [f"Page {page['page']}" for page in st.session_state.pages_data]
    selected_page_idx = st.selectbox(
        "Select a page to process:",
        range(len(page_numbers)),
        format_func=lambda x: page_numbers[x],
    )

    # Display the selected page
    if selected_page_idx < len(st.session_state.pages_data):
        page_data = st.session_state.pages_data[selected_page_idx]

        st.markdown(f"### {page_numbers[selected_page_idx]}")

        # Show image preview of the page
        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(
                page_data["image"],
                caption=f"Page {page_data['page']}",
                use_container_width=True,
            )

        with col2:
            # Button to extract text from this page
            if st.button("Extract Text"):
                with st.spinner("Extracting text from page..."):
                    # Extract text using OpenAI Vision
                    extracted_text = extract_text_from_image_openai(page_data["image"])
                    # Save the extracted text in the pages_data
                    st.session_state.pages_data[selected_page_idx][
                        "text"
                    ] = extracted_text
                    st.rerun()

            # Show extracted text if available
            if page_data.get("text"):
                st.markdown("**Extracted Text:**")
                st.text_area(
                    f"Text from Page {page_data['page']}",
                    page_data["text"],
                    height=200,
                )

                # Generate audio for text pages
                if st.button(f"Generate Audio for {page_numbers[selected_page_idx]}"):
                    with st.spinner("Generating audio"):
                        audio_data = text_to_speech(page_data["text"], voice)
                        if audio_data:
                            # Save audio data to a temporary file
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".wav"
                            ) as temp_audio_file:
                                # Write the audio data to the file
                                audio_data.seek(0)
                                temp_audio_file.write(audio_data.read())
                                audio_file_path = temp_audio_file.name

                            # Reset audio data to beginning for display
                            audio_data.seek(0)

                            # Display audio player
                            st.audio(audio_data, format="audio/mp3")

                            # Store audio file path in session state
                            st.session_state.audio_files[selected_page_idx] = (
                                audio_file_path
                            )

                            # Create a download link
                            download_filename = f"page_{page_data['page']}.mp3"
                            st.download_button(
                                label="Download MP3",
                                data=audio_data,
                                file_name=download_filename,
                                mime="audio/mp3",
                            )

    # Add a batch processing section
    st.markdown("---")
    st.markdown("### Batch Processing")

    # Page range selection
    total_pages = len(st.session_state.pages_data)
    col1, col2 = st.columns(2)

    with col1:
        start_page = st.number_input(
            "Start page:", min_value=1, max_value=total_pages, value=1
        )

    with col2:
        end_page = st.number_input(
            "End page:",
            min_value=start_page,
            max_value=total_pages,
            value=total_pages,
        )

    # If batch processing is in progress, show progress
    if st.session_state.batch_processing:
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_page_info = st.empty()  # Container for current page info

        # Continue processing from where we left off
        start_idx = st.session_state.batch_page_index
        # Process all pages within selected range
        end_idx = end_page

        # Only process pages within selected range
        start_idx = max(start_idx, start_page - 1)
        end_idx = min(end_idx, end_page)

        # Update progress bar
        if st.session_state.batch_total > 0:
            progress_percentage = (
                st.session_state.batch_progress / st.session_state.batch_total
            )
            progress_bar.progress(progress_percentage)
            status_text.text(
                f"Processing {st.session_state.batch_progress} of {st.session_state.batch_total} pages..."
            )

        # Process current batch
        pages_to_process = st.session_state.pages_data[start_idx:end_idx]

        for i, page in enumerate(pages_to_process):
            current_idx = start_idx + i
            page_num = page["page"]

            status_text.text(f"Processing Page {page_num}...")
            current_page_info.text(f"Current page: {page_num} of {end_page}")

            # Extract text if not already available
            if not page.get("text"):
                page_text = extract_text_from_image_openai(page["image"])
                st.session_state.pages_data[current_idx]["text"] = page_text

            # Clear any previous content from page info area
            current_page_info.empty()
            current_page_info.text(f"Current page: {page_num} of {end_page}")

            # Generate audio if not already available
            if page.get("text") and current_idx not in st.session_state.audio_files:
                audio_data = text_to_speech(page["text"], voice)
                if audio_data:
                    # Save audio data to a temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_audio_file:
                        # Write the audio data to the file
                        audio_data.seek(0)
                        temp_audio_file.write(audio_data.read())
                        audio_file_path = temp_audio_file.name

                    # Store audio file path in session state
                    st.session_state.audio_files[current_idx] = audio_file_path

            # Update progress
            st.session_state.batch_progress += 1
            progress_percentage = (
                st.session_state.batch_progress / st.session_state.batch_total
            )
            progress_bar.progress(progress_percentage)

            # Force garbage collection to free memory
            gc.collect()

        # Update batch_page_index for next batch
        st.session_state.batch_page_index = end_idx

        # Check if we're done
        if st.session_state.batch_page_index >= end_page:
            st.session_state.batch_processing = False
            status_text.text("Batch processing complete!")
            current_page_info.empty()  # Clear the info area

            # Summary of processed pages
            st.success(
                f"Successfully processed {st.session_state.batch_progress} pages."
            )

            # Auto-merge all audio files when processing is complete
            with st.spinner("Merging all audio files..."):
                merged_audio = merge_audio_files(st.session_state.audio_files)
                if merged_audio:
                    st.session_state.merged_audio = merged_audio
                    st.success("Audio files merged successfully!")
                    st.rerun()
    else:
        # Option to start batch processing
        if st.button("Start Batch Processing"):
            # Initialize batch process
            pages_to_process = list(range(start_page - 1, end_page))
            st.session_state.batch_page_index = start_page - 1
            st.session_state.batch_total = len(pages_to_process)
            st.session_state.batch_progress = 0
            st.session_state.batch_processing = True
            st.rerun()

    # Show a summary of processed pages
    if len(st.session_state.audio_files) > 0:
        st.markdown("---")
        st.markdown("### Processed Pages Summary")

        # Display in a more compact form
        processed_pages = sorted(list(st.session_state.audio_files.keys()))

        st.success(f"Audio generated for {len(processed_pages)} pages")

        # Display merged audio if available
        if st.session_state.merged_audio:
            st.markdown("### Complete PDF Audio")
            st.audio(st.session_state.merged_audio, format="audio/mp3")

            # Get filename from the PDF
            pdf_filename = os.path.basename(st.session_state.pdf_path).split(".")[0]
            download_filename = f"{pdf_filename}_complete.mp3"

            # Create a download button for the merged audio
            st.download_button(
                label="Download Complete PDF Audio",
                data=st.session_state.merged_audio,
                file_name=download_filename,
                mime="audio/mp3",
                key="merged_audio_download",
            )
        else:
            # Add option to merge all audio files manually if it wasn't done automatically
            if st.button("Merge All Audio Files into One"):
                with st.spinner("Merging audio files..."):
                    merged_audio = merge_audio_files(st.session_state.audio_files)
                    if merged_audio:
                        st.session_state.merged_audio = merged_audio
                        st.success("Audio files merged successfully!")
                        st.rerun()
else:
    st.info("Please upload a PDF file to begin.")

# Add footer
st.markdown("---")

# Run garbage collection to free memory
gc.collect()
