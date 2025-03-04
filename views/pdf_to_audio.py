import os
import tempfile
import gc  # Garbage collection

import streamlit as st
from tools.pdf_tools import extract_text_from_pdf, text_to_speech, merge_audio_files

st.set_page_config(
    page_title="PDF to Audio Converter",
    page_icon="📄",
    layout="wide",
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

    # Register cleanup function to run when the app exits
    import atexit

    atexit.register(cleanup)


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
            st.session_state.pages_data = extract_text_from_pdf(
                st.session_state.pdf_path
            )
            # Reset other session state variables
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

    # Create a dropdown to select the page
    page_numbers = [f"Page {page['page']}" for page in st.session_state.pages_data]
    selected_page_idx = st.selectbox(
        "Select a page to process:",
        range(len(page_numbers)),
        format_func=lambda x: page_numbers[x],
    )

    # Display the selected page
    page_data = st.session_state.pages_data[selected_page_idx]

    st.markdown(f"### {page_numbers[selected_page_idx]}")

    # Handle page based on type
    if page_data["text"]:
        st.markdown("**Extracted Text:**")
        st.text_area(
            f"Text from {page_numbers[selected_page_idx]}",
            page_data["text"],
            height=200,
        )

        # Generate audio for text pages
        if st.button(f"Generate Audio for {page_numbers[selected_page_idx]}"):
            with st.spinner("Generating audio with OpenAI TTS..."):
                audio_data = text_to_speech(page_data["text"], voice)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")

                    # Store audio data in session state
                    st.session_state.audio_files[selected_page_idx] = audio_data

                    # Create a download link
                    download_filename = f"page_{page_data['page']}.mp3"
                    st.download_button(
                        label="Download MP3",
                        data=audio_data,
                        file_name=download_filename,
                        mime="audio/mp3",
                    )
                    # st.markdown(
                    #     get_binary_file_downloader_html(
                    #         audio_data, "Download MP3", download_filename
                    #     ),
                    #     unsafe_allow_html=True,
                    # )

    # Add a batch processing section
    st.markdown("---")
    st.markdown("### Batch Processing")

    # Display batch processing options
    col1, col2 = st.columns(2)

    # Batch size selection
    batch_size_options = [1, 2, 3, 5, 10]
    with col1:
        batch_size = st.selectbox("Pages per batch:", batch_size_options, index=1)

    # Page range selection
    with col2:
        total_pages = len(st.session_state.pages_data)
        start_page = st.number_input(
            "Start page:", min_value=1, max_value=total_pages, value=1
        )
        end_page = st.number_input(
            "End page:",
            min_value=start_page,
            max_value=total_pages,
            value=min(start_page + batch_size - 1, total_pages),
        )

    # If batch processing is in progress, show progress
    if st.session_state.batch_processing:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Continue processing from where we left off
        start_idx = st.session_state.batch_page_index
        end_idx = min(start_idx + batch_size, len(st.session_state.pages_data))

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

            if page["text"]:
                if current_idx not in st.session_state.audio_files:
                    audio_data = text_to_speech(page["text"], voice)
                    if audio_data:
                        # Store audio data in session state
                        st.session_state.audio_files[current_idx] = audio_data

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

            # Summary of processed pages
            st.success(
                f"Successfully processed {st.session_state.batch_progress} pages."
            )

            # Add a refresh button to see results
            if st.button("Refresh to see results"):
                st.rerun()
        else:
            # Not done yet, add continue button
            if st.button("Process Next Batch"):
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
        pages_text = ", ".join(
            [
                f"Page {st.session_state.pages_data[idx]['page']}"
                for idx in processed_pages
            ]
        )
        st.success(f"Audio generated for {len(processed_pages)} pages: {pages_text}")

        # Add option to merge all audio files
        if st.button("Merge All Audio Files into One"):
            with st.spinner("Merging audio files..."):
                merged_audio = merge_audio_files(st.session_state.audio_files)
                if merged_audio:
                    st.session_state.merged_audio = merged_audio
                    st.success("Audio files merged successfully!")
                    st.rerun()

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
    st.info("Please upload a PDF file to begin.")

# Add footer
st.markdown("---")
st.markdown("PDF to Speech Converter | Built with Streamlit, PyPDF, and OpenAI")

# Run garbage collection to free memory
gc.collect()
