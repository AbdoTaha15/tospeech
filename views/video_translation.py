import os
import tempfile
import streamlit as st
import gc  # Garbage collection
from pydub import AudioSegment
import time
import numpy as np
from tools.video_tools import (
    extract_audio_from_video,
    translate_audio,
    merge_video_with_audio,
    split_long_audio,
    adjust_audio_speed,
)

st.set_page_config(
    page_title="Video Translation Tool",
    page_icon="🎥",
    layout="wide",
)

# Initialize session state variables
if (
    "video_translation_initialized" not in st.session_state
    or not st.session_state.video_translation_initialized
):
    st.session_state.video_path = None
    st.session_state.audio_path = None
    st.session_state.translated_audio_path = None
    st.session_state.output_video_path = None
    st.session_state.processing_step = None
    st.session_state.transcription = None
    st.session_state.translation = None
    st.session_state.video_metadata = None
    st.session_state.debug_info = None
    st.session_state.video_translation_initialized = True

    # Clean up temporary files when app exits
    def cleanup():
        for attr in [
            "video_path",
            "audio_path",
            "translated_audio_path",
            "output_video_path",
        ]:
            if (
                attr in st.session_state
                and st.session_state[attr]
                and os.path.exists(st.session_state[attr])
            ):
                try:
                    os.unlink(st.session_state[attr])
                except Exception:
                    pass

    # Register cleanup function
    import atexit

    atexit.register(cleanup)

# UI Layout
st.title("🎥 Video Translation Tool")
st.markdown("Upload a video and translate its audio to your desired language.")

# File uploader for video
uploaded_file = st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi", "mkv"]
)

# Main processing section
if uploaded_file:
    # Save the uploaded file
    file_changed = False

    if st.session_state.video_path is None:
        file_changed = True
    else:
        # Check if the file has changed by comparing sizes or other metadata
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            new_video_path = temp_file.name
            uploaded_file.seek(0)  # Reset file pointer

            if os.path.getsize(new_video_path) != os.path.getsize(
                st.session_state.video_path
            ):
                file_changed = True
                # Clean up old files
                for attr in [
                    "video_path",
                    "audio_path",
                    "translated_audio_path",
                    "output_video_path",
                ]:
                    if (
                        attr in st.session_state
                        and st.session_state[attr]
                        and os.path.exists(st.session_state[attr])
                    ):
                        try:
                            os.unlink(st.session_state[attr])
                        except Exception:
                            pass
                st.session_state.video_path = new_video_path
            else:
                os.unlink(new_video_path)

    # If new file is uploaded, process it
    if file_changed:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            st.session_state.video_path = temp_file.name

        # Reset related session state
        st.session_state.audio_path = None
        st.session_state.transcription = None
        st.session_state.translation = None
        st.session_state.translated_audio_path = None
        st.session_state.output_video_path = None
        st.session_state.processing_step = None
        st.session_state.debug_info = None

    # Target language selection
    # target_language_options = {
    #     "English": "English",
    #     "Arabic": "Arabic",
    #     "French": "French",
    #     "German": "German",
    #     "Spanish": "Spanish",
    #     "Italian": "Italian",
    #     "Japanese": "Japanese",
    #     "Korean": "Korean",
    #     "Portuguese": "Portuguese",
    #     "Russian": "Russian",
    #     "Chinese": "Chinese (Mandarin)",
    # }
    # target_language = st.selectbox("Translate to", list(target_language_options.keys()))
    target_lang_full = st.text_input(
        "Write the full name of the target language", "Arabic"
    )

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
        handle_long_videos = st.checkbox(
            "Handle long videos by splitting audio", value=True
        )
        max_segment_duration = st.slider(
            "Maximum segment duration (seconds)", 10, 120, 25
        )
        show_debug_info = st.checkbox("Show debugging information", value=False)

    # Process video button
    if st.button("Start Translation Process"):
        # Step 1: Extract audio
        if not st.session_state.audio_path:
            st.session_state.audio_path = extract_audio_from_video(
                st.session_state.video_path
            )
            if not st.session_state.audio_path:
                st.error(
                    "Failed to extract audio from video. Please try a different video file."
                )
                st.stop()
            st.session_state.processing_step = "audio_extracted"

        # Step 2: Translate audio directly (combines transcription, translation, and TTS)
        if (
            st.session_state.processing_step == "audio_extracted"
            and not st.session_state.translated_audio_path
        ):
            # Check if we need to split long audio
            audio_segments = [st.session_state.audio_path]
            if handle_long_videos:
                audio_segments = split_long_audio(
                    st.session_state.audio_path, max_segment_duration
                )
                if len(audio_segments) > 1:
                    st.info(
                        f"Audio split into {len(audio_segments)} segments for processing"
                    )

            # Process each segment
            if len(audio_segments) == 1:
                # Single segment processing
                (
                    st.session_state.transcription,
                    st.session_state.translation,
                    st.session_state.translated_audio_path,
                ) = translate_audio(
                    st.session_state.audio_path, target_lang_full, voice
                )
                st.write(st.session_state.transcription)
                st.write(st.session_state.translation)
            else:
                # Multiple segment processing
                segment_translations = []
                segment_audio_paths = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, segment_path in enumerate(audio_segments):
                    status_text.text(
                        f"Processing segment {i+1} of {len(audio_segments)}..."
                    )
                    _, segment_translation, segment_audio_path = translate_audio(
                        segment_path, target_lang_full, voice
                    )

                    if segment_translation and segment_audio_path:
                        segment_translations.append(segment_translation)
                        segment_audio_paths.append(segment_audio_path)

                    # Update progress
                    progress_bar.progress((i + 1) / len(audio_segments))

                # Combine results
                st.session_state.translation = "\n\n".join(segment_translations)

                # Merge audio segments
                with st.spinner("Merging translated audio segments..."):
                    merged_audio = AudioSegment.empty()
                    for path in segment_audio_paths:
                        segment = AudioSegment.from_file(path)
                        merged_audio += segment

                    # Save merged audio
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_audio:
                        merged_audio.export(temp_audio.name, format="wav")
                        st.session_state.translated_audio_path = temp_audio.name

                st.success(f"Successfully processed {len(audio_segments)} segments")

            if not st.session_state.translated_audio_path:
                st.error(
                    "Failed to translate audio. Please try again or use a different video."
                )
                st.stop()

            st.session_state.processing_step = "translated"

        # Step 3: Merge video with translated audio
        if (
            st.session_state.processing_step == "translated"
            and not st.session_state.output_video_path
        ):
            # Always load the translated audio
            translated_audio = AudioSegment.from_file(
                st.session_state.translated_audio_path
            )

            # Load the original audio to get its duration
            if st.session_state.audio_path:
                original_audio = AudioSegment.from_file(st.session_state.audio_path)
                original_duration = len(original_audio)
                translated_duration = len(translated_audio)

                # Adjust speed of translated audio to match original duration if they differ significantly
                speed_ratio = translated_duration / original_duration

                if speed_ratio != 1.0:
                    st.info(
                        f"Adjusting translated audio speed by {speed_ratio:.2f}x to match video length"
                    )

                    # Use our new Python function instead of ffmpeg command
                    speed_adjusted_path = adjust_audio_speed(
                        st.session_state.translated_audio_path, speed_ratio
                    )

                    # Update the translated audio path to use the speed-adjusted version
                    translated_audio = AudioSegment.from_file(speed_adjusted_path)
                    # Update the session state to use the adjusted audio
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_audio:
                        translated_audio.export(temp_audio.name, format="wav")
                        st.session_state.translated_audio_path = temp_audio.name

            # # If preserving original audio at low volume is selected
            # if preserve_original_audio and st.session_state.audio_path:
            #     with st.spinner("Mixing original and translated audio..."):
            #         # Load both audio files
            #         original_audio = AudioSegment.from_file(st.session_state.audio_path)
            #         translated_audio = AudioSegment.from_file(
            #             st.session_state.translated_audio_path
            #         )

            #         # Get durations
            #         original_duration = len(original_audio)
            #         translated_duration = len(translated_audio)

            #         # Match volumes - reduce original volume substantially
            #         original_audio = original_audio - 20  # Reduce by 20 dB

            #         # Ensure both audio segments are the same length
            #         if translated_duration < original_duration:
            #             # Add silence to end of translated audio
            #             silence = AudioSegment.silent(
            #                 duration=original_duration - translated_duration
            #             )
            #             translated_audio = translated_audio + silence
            #         elif original_duration < translated_duration:
            #             # Add silence to end of original audio
            #             silence = AudioSegment.silent(
            #                 duration=translated_duration - original_duration
            #             )
            #             original_audio = original_audio + silence

            #         # Mix the audio tracks with appropriate balance
            #         # Use overlay with gain_during_overlay to control the mix better
            #         mixed_audio = translated_audio.overlay(
            #             original_audio,
            #             position=0,
            #             gain_during_overlay=-5,  # Additional reduction during overlay
            #         )

            #         # Apply a slight normalization to prevent clipping
            #         peak_amplitude = max(
            #             abs(np.array(mixed_audio.get_array_of_samples()).max()),
            #             abs(np.array(mixed_audio.get_array_of_samples()).min()),
            #         )

            #         # Normalize only if needed to prevent clipping
            #         if peak_amplitude > 32700:  # Close to 16-bit max (32767)
            #             normalized_audio = (
            #                 mixed_audio - 3
            #             )  # Reduce by 3dB to prevent clipping
            #         else:
            #             normalized_audio = mixed_audio

            #         # Save the mixed audio with high quality settings
            #         with tempfile.NamedTemporaryFile(
            #             delete=False, suffix=".wav"
            #         ) as temp_mixed:
            #             normalized_audio.export(
            #                 temp_mixed.name,
            #                 format="wav",
            #                 parameters=["-q:a", "0"],  # Use highest quality
            #             )
            #             # Replace the translated audio path with the mixed version
            #             st.session_state.translated_audio_path = temp_mixed.name

            # Merge with video
            start_time = time.time()
            st.session_state.output_video_path = merge_video_with_audio(
                st.session_state.video_path, st.session_state.translated_audio_path
            )
            processing_time = time.time() - start_time
            st.session_state.debug_info = f"{st.session_state.debug_info}\nVideo processing time: {processing_time:.1f}s"

            if not st.session_state.output_video_path:
                st.error(
                    "Failed to merge video with translated audio. Please try again."
                )
                st.stop()

            st.session_state.processing_step = "completed"

    # Display debug information if enabled
    if show_debug_info and st.session_state.debug_info:
        with st.expander("Debug Information"):
            st.code(st.session_state.debug_info)

            if st.session_state.video_metadata:
                st.subheader("Video Metadata")
                st.json(st.session_state.video_metadata)

    # Display results based on current step
    if st.session_state.translation and st.session_state.processing_step in [
        "translated",
        "completed",
    ]:
        st.subheader(f"Translation to {target_lang_full}")
        st.text_area("Translated content", st.session_state.translation, height=150)

    if st.session_state.translated_audio_path and st.session_state.processing_step in [
        "translated",
        "completed",
    ]:
        st.subheader("Translated Audio")
        with open(st.session_state.translated_audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")

    if st.session_state.processing_step == "completed":
        st.subheader("Translated Video")
        # Create a video player for the output video
        with open(st.session_state.output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

        # Create download button for the translated video
        with open(st.session_state.output_video_path, "rb") as file:
            video_name = uploaded_file.name.split(".")[0]
            target_lang_short = target_lang_full.lower()
            st.download_button(
                label="Download Translated Video",
                data=file,
                file_name=f"{video_name}_translated_to_{target_lang_short}.mp4",
                mime="video/mp4",
            )

        # Option to restart
        if st.button("Start New Translation"):
            # Clean up files
            for attr in [
                "video_path",
                "audio_path",
                "translated_audio_path",
                "output_video_path",
            ]:
                if (
                    attr in st.session_state
                    and st.session_state[attr]
                    and os.path.exists(st.session_state[attr])
                ):
                    try:
                        os.unlink(st.session_state[attr])
                    except Exception:
                        pass

            # Reset session state
            st.session_state.video_path = None
            st.session_state.audio_path = None
            st.session_state.transcription = None
            st.session_state.translation = None
            st.session_state.translated_audio_path = None
            st.session_state.output_video_path = None
            st.session_state.processing_step = None
            st.session_state.debug_info = None


else:
    st.info("Please upload a video file to begin the translation process.")

# Add footer
st.markdown("---")
st.markdown("Video Translation Tool | Built with Streamlit, MoviePy, and OpenAI")

# Force garbage collection to free memory
gc.collect()
