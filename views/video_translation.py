import os
import tempfile
import streamlit as st
from pathlib import Path
import io
import gc  # Garbage collection
from moviepy import VideoFileClip, AudioFileClip
import openai
import base64
from pydub import AudioSegment
import subprocess
import json
import time
import shlex

# Initialize OpenAI client from environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


# Define helper functions for video translation process
def get_video_metadata(video_path):
    """Extract metadata from video file using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if result.stderr:
                st.error(f"ffprobe error: {result.stderr}")
            return None

        metadata = json.loads(result.stdout)
        return metadata
    except Exception as e:
        st.warning(f"Could not extract video metadata: {str(e)}")
        return None


def extract_audio_using_ffmpeg(video_path):
    """Extract audio using direct ffmpeg command for more control"""
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name

        # Set up ffmpeg command with specific parameters for better compatibility
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM format
            "-ar",
            "44100",  # Sample rate
            "-ac",
            "1",  # Mono
            "-y",  # Overwrite output
            audio_path,
        ]

        # Log the command for debugging
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
        st.session_state.debug_info = f"Running: {cmd_str}"

        # Run the command
        process = subprocess.run(cmd, capture_output=True, text=True)

        # Check if successful
        if process.returncode != 0:
            error_message = process.stderr
            st.error(f"ffmpeg error: {error_message}")
            return None

        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            st.error("Audio extraction failed: output file is empty or not created")
            return None

        return audio_path
    except Exception as e:
        st.error(f"Error in audio extraction: {str(e)}")
        return None


def extract_audio_from_video(video_path):
    """Extract audio track from a video file with fallbacks"""
    try:
        with st.spinner("Extracting audio from video..."):
            # Get video metadata to determine optimal audio extraction parameters
            metadata = get_video_metadata(video_path)
            if metadata:
                # Store metadata for debugging
                st.session_state.video_metadata = metadata
                duration = metadata.get("format", {}).get("duration")
                if duration:
                    st.info(f"Processing video: {float(duration):.1f}s duration")

            # Try direct ffmpeg extraction first (more reliable for some formats)
            audio_path = extract_audio_using_ffmpeg(video_path)
            if (
                audio_path
                and os.path.exists(audio_path)
                and os.path.getsize(audio_path) > 0
            ):
                return audio_path

            # Fall back to MoviePy as an alternative
            st.warning("Direct ffmpeg extraction failed. Trying alternative method...")

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_path = temp_audio.name

            # Extract the audio using MoviePy with optimized parameters
            video = VideoFileClip(video_path)
            if video.audio is None:
                st.error("No audio track found in the video file")
                video.close()
                return None

            video.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",  # Ensure WAV format with PCM encoding
                ffmpeg_params=["-ac", "1", "-ar", "44100"],  # Mono audio at 44.1kHz
                verbose=False,
                logger=None,
            )
            video.close()

            # Verify audio file was created successfully
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                st.error(
                    "Failed to extract audio from video - audio track might be missing or corrupted"
                )
                return None

            return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None


def handle_portrait_video(video_path):
    """Check if video is in portrait mode and handle rotation if needed"""
    try:
        metadata = get_video_metadata(video_path)
        if not metadata:
            return video_path

        # Extract video stream information
        video_streams = [
            s for s in metadata.get("streams", []) if s.get("codec_type") == "video"
        ]
        if not video_streams:
            return video_path

        video_stream = video_streams[0]
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Check if portrait orientation
        if height > width:
            st.info(f"Portrait video detected: {width}x{height}")

        # Look for rotation metadata
        rotation = 0
        tags = video_stream.get("tags", {})
        if "rotate" in tags:
            rotation = int(tags["rotate"])
            st.info(f"Video has rotation metadata: {rotation} degrees")

        # No need to actually rotate - MoviePy handles this automatically
        # Just storing the information for potential debugging
        st.session_state.video_orientation = {
            "width": width,
            "height": height,
            "rotation": rotation,
            "is_portrait": height > width,
        }

        return video_path
    except Exception as e:
        st.warning(f"Could not check video orientation: {str(e)}")
        return video_path


def translate_audio(audio_path, target_language, voice="alloy"):
    """Translate audio directly to target language using GPT-4o Audio Preview"""
    try:
        # Check audio file size before processing
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        if audio_size_mb > 25:
            st.warning(
                f"Audio file size ({audio_size_mb:.1f} MB) exceeds 25MB limit. Processing may fail or need to be broken into segments."
            )

        with st.spinner(f"Translating audio to {target_language}..."):
            # Read audio file and encode it as base64
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                encoded_audio = base64.b64encode(audio_data).decode("utf-8")

                # Using GPT-4o Audio Preview for direct translation
                completion = client.chat.completions.create(
                    model="gpt-4o-mini-audio-preview",
                    modalities=["text", "audio"],
                    audio={"voice": voice, "format": "wav"},
                    messages=[
                        {
                            "role": "developer",
                            "content": f"You are a professional translator. Translate the input audio to {target_language}. Maintain the original tone, style, and emotion. Focus only on translation without adding any comments. Try to keep the translation as close to the original length as possible.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Listen to this audio and translate it to {target_language}. Then generate audio of the translation in a natural-sounding {target_language} voice.",
                                },
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded_audio,
                                        "format": "wav",
                                    },
                                },
                            ],
                        },
                    ],
                )

                # Extract the transcription and translation
                transcription = "Original audio transcription is included in the direct translation process."
                translation = completion.choices[0].message.content

                # Extract the audio data and save to a temporary file
                wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_audio:
                    temp_audio.write(wav_bytes)
                    translated_audio_path = temp_audio.name

                return transcription, translation, translated_audio_path
    except Exception as e:
        st.error(f"Error translating audio: {str(e)}")
        return None, None, None


def merge_video_with_audio(video_path, audio_path):
    """Merge video with a new audio track"""
    try:
        with st.spinner("Merging video with translated audio..."):
            # Get video metadata to use correct parameters
            metadata = get_video_metadata(video_path)
            if metadata and metadata.get("streams"):
                video_streams = [
                    s
                    for s in metadata.get("streams", [])
                    if s.get("codec_type") == "video"
                ]
                if video_streams:
                    video_stream = video_streams[0]
                    width = int(video_stream.get("width", 0))
                    height = int(video_stream.get("height", 0))
                    fps = eval(video_stream.get("r_frame_rate", "30/1"))
                    st.info(f"Video properties: {width}x{height} @ {fps}fps")

            # Create output file path
            output_path = tempfile.mktemp(suffix=".mp4")

            # Try direct ffmpeg approach first for better control
            try:
                # Load the audio file information
                audio_info = AudioSegment.from_file(audio_path)
                audio_duration_ms = len(audio_info)

                # Build the ffmpeg command
                cmd = [
                    "ffmpeg",
                    "-i",
                    video_path,  # Input video
                    "-i",
                    audio_path,  # Input audio
                    "-map",
                    "0:v:0",  # Use first video stream from first input
                    "-map",
                    "1:a:0",  # Use first audio stream from second input
                    "-c:v",
                    "copy",  # Copy video stream (no re-encoding)
                    "-c:a",
                    "aac",  # AAC audio codec
                    "-shortest",  # End when shortest input ends
                    "-y",  # Overwrite output
                    output_path,
                ]

                # Run the command
                process = subprocess.run(cmd, capture_output=True, text=True)

                if process.returncode == 0:
                    # Verify the output file exists and has content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
                    else:
                        st.warning(
                            "Output video file is empty. Falling back to MoviePy..."
                        )
                else:
                    st.warning(
                        f"ffmpeg merge failed: {process.stderr}. Falling back to MoviePy..."
                    )
            except Exception as e:
                st.warning(
                    f"Direct ffmpeg merge failed: {str(e)}. Falling back to MoviePy..."
                )

            # Fall back to MoviePy if ffmpeg direct approach fails
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)

            # If the audio is shorter than the video, we need to handle that
            if audio.duration < video.duration:
                st.warning(
                    f"The translated audio ({audio.duration:.1f}s) is shorter than the video ({video.duration:.1f}s). The remaining part will be silent."
                )

            # Set the audio of the video
            video = video.set_audio(audio)

            # Write the result with parameters optimized based on original video
            video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=None,  # Let ffmpeg choose appropriate bitrate
                fps=video.fps,
                preset="medium",  # Balance between quality and encoding speed
                threads=2,  # Use multiple threads for faster encoding
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                verbose=False,
                logger=None,
            )

            # Close clips to release resources
            video.close()
            audio.close()

            return output_path
    except Exception as e:
        st.error(f"Error merging video with audio: {str(e)}")
        return None


def split_long_audio(audio_path, max_duration=25):
    """Split long audio files into smaller chunks for processing"""
    try:
        # Check if we need to split
        audio = AudioSegment.from_file(audio_path)
        audio_duration_seconds = len(audio) / 1000

        if audio_duration_seconds <= max_duration:
            return [audio_path]

        # Calculate number of segments needed
        num_segments = int(audio_duration_seconds / max_duration) + 1
        segment_duration = len(audio) / num_segments

        # Split the audio
        segments = []
        for i in range(num_segments):
            start_pos = int(i * segment_duration)
            end_pos = int((i + 1) * segment_duration)

            # Handle last segment
            if i == num_segments - 1:
                end_pos = len(audio)

            segment = audio[start_pos:end_pos]

            # Save segment to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_segment:
                segment_path = temp_segment.name
                segment.export(segment_path, format="wav")
                segments.append(segment_path)

        return segments
    except Exception as e:
        st.error(f"Error splitting audio: {str(e)}")
        return [audio_path]  # Fall back to original file


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

        # Check for portrait orientation and handle if needed
        st.session_state.video_path = handle_portrait_video(st.session_state.video_path)

        # Reset related session state
        st.session_state.audio_path = None
        st.session_state.transcription = None
        st.session_state.translation = None
        st.session_state.translated_audio_path = None
        st.session_state.output_video_path = None
        st.session_state.processing_step = None
        st.session_state.debug_info = None

    # Display the uploaded video
    st.video(uploaded_file)

    # Target language selection
    target_language_options = {
        "English": "English",
        "Arabic": "Arabic",
        "French": "French",
        "German": "German",
        "Spanish": "Spanish",
        "Italian": "Italian",
        "Japanese": "Japanese",
        "Korean": "Korean",
        "Portuguese": "Portuguese",
        "Russian": "Russian",
        "Chinese": "Chinese (Mandarin)",
    }
    target_language = st.selectbox("Translate to", list(target_language_options.keys()))
    target_lang_full = target_language_options[target_language]

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
        preserve_original_audio = st.checkbox(
            "Include original audio at low volume", value=False
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

                    # Create a temporary file for the speed-adjusted audio
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_speed:
                        speed_adjusted_path = temp_speed.name

                    # Use ffmpeg to change the speed without changing the pitch
                    cmd = [
                        "ffmpeg",
                        "-i",
                        st.session_state.translated_audio_path,
                        "-filter:a",
                        f"atempo={speed_ratio}",
                        "-y",
                        speed_adjusted_path,
                    ]

                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        # Update the translated audio path to use the speed-adjusted version
                        translated_audio = AudioSegment.from_file(speed_adjusted_path)
                        # Update the session state to use the adjusted audio
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as temp_audio:
                            translated_audio.export(temp_audio.name, format="wav")
                            st.session_state.translated_audio_path = temp_audio.name
                    except Exception as e:
                        st.warning(f"Could not adjust audio speed: {str(e)}")

            # If preserving original audio at low volume is selected
            if preserve_original_audio and st.session_state.audio_path:
                with st.spinner("Mixing original and translated audio..."):
                    # Reduce volume of original audio
                    original_audio = original_audio - 15  # Reduce by 15 dB

                    # Make sure translated audio is at least as long as original
                    if len(translated_audio) < len(original_audio):
                        # Add silence to match lengths
                        silence = AudioSegment.silent(
                            duration=len(original_audio) - len(translated_audio)
                        )
                        translated_audio = translated_audio + silence

                    # Mix the audio tracks
                    mixed_audio = translated_audio.overlay(original_audio)

                    # Save the mixed audio
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_mixed:
                        mixed_audio.export(temp_mixed.name, format="wav")
                        # Replace the translated audio path with the mixed version
                        st.session_state.translated_audio_path = temp_mixed.name

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
        st.subheader(f"Translation to {target_language}")
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
            target_lang_short = target_language.lower()
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
