import os
import gc
import tempfile
from moviepy import VideoFileClip, AudioFileClip
import openai
from pydub import AudioSegment
import streamlit as st
import numpy as np
import pyrubberband as pyrb

# Initialize OpenAI client from environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Define helper functions for video translation process
def get_video_metadata(video_path):
    """Extract metadata from video file using MoviePy"""
    try:
        with VideoFileClip(video_path) as video_clip:
            # Get video properties
            metadata = {
                "format": {
                    "duration": video_clip.duration,
                    "size": os.path.getsize(video_path),
                    "filename": os.path.basename(video_path),
                },
                "streams": [
                    {
                        "codec_type": "video",
                        "width": video_clip.size[0],
                        "height": video_clip.size[1],
                        "r_frame_rate": f"{video_clip.fps}/1",
                        "tags": {},
                    }
                ],
            }

            # Check for audio stream
            if video_clip.audio is not None:
                metadata["streams"].append(
                    {
                        "codec_type": "audio",
                        "sample_rate": video_clip.audio.fps,
                        "channels": video_clip.audio.nchannels,
                    }
                )

            return metadata
    except Exception as e:
        st.warning(f"Could not extract video metadata: {str(e)}")
        return None


def extract_audio_from_video(video_path):
    """Extract audio track from a video file using MoviePy"""
    try:
        with st.spinner("Extracting audio from video..."):
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_path = temp_audio.name

            # Extract the audio using MoviePy
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    st.error("No audio track found in the video file")
                    return None

                # Extract audio with desired parameters
                video.audio.write_audiofile(
                    audio_path,
                    fps=44100,  # Sample rate
                    nbytes=2,  # 16-bit
                    codec="pcm_s16le",
                    ffmpeg_params=["-ac", "1"],  # Force mono
                    logger=None,
                )

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


def translate_audio(audio_path, target_language, voice="alloy"):
    """Translate audio directly to target language using GPT-4o Audio Preview"""
    try:
        # Check audio file size before processing
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        if audio_size_mb > 20:
            st.warning(
                f"Audio file size ({audio_size_mb:.1f} MB) exceeds 20MB limit. Processing may fail or need to be broken into segments."
            )

        with st.spinner(f"Translating audio to {target_language}..."):
            # Read audio file and encode it as base64
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                ).text

                # Using GPT-4o Audio Preview for direct translation
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "developer",
                            "content": f"""
Role: You are an expert professional translator. Your task is to translate the complete text into {target_language}.

Task:
    - Translate the entire provided text to {target_language}.
    - Ensure that no part of the original text is omitted or summarized; every word, punctuation, and nuance must be preserved.
    - Maintain the full context and meaning of the input text.
    - Use the most natural and fluent language style for the target audience.
    - Ensure that the translation is grammatically correct and culturally appropriate.
    - Use the most suitable dialect or variant of the target language, if applicable.
    
Expected Output:
    - Provide the translated text in the same format as the input.
    - Do not add any additional comments, explanations, or translator notes.
""",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": transcription,
                                },
                            ],
                        },
                    ],
                )
                translation = completion.choices[0].message.content

                with client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice=voice,
                    input=translation,
                ) as response:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_audio:
                        response.stream_to_file(temp_audio.name)
                        translated_audio_path = temp_audio.name

                return transcription, translation, translated_audio_path
    except Exception as e:
        st.error(f"Error translating audio: {str(e)}")
        return None, None, None


def merge_video_with_audio(video_path, audio_path):
    """Merge video with a new audio track using MoviePy"""
    try:
        with st.spinner("Merging video with translated audio..."):
            # Create output file path
            output_path = tempfile.mktemp(suffix=".mp4")

            # Load video and audio clips
            with VideoFileClip(video_path) as video, AudioFileClip(audio_path) as audio:
                # Set the audio of the video
                video: VideoFileClip = video.with_audio(audio)

                # Write the result with parameters optimized based on original video
                video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate=None,  # Let MoviePy choose appropriate bitrate
                    fps=video.fps,
                    preset="medium",  # Balance between quality and encoding speed
                    threads=2,  # Use multiple threads for faster encoding
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    logger=None,
                )

            return output_path
    except Exception as e:
        st.error(f"Error merging video with audio: {str(e)}")
        return None


def split_long_audio(audio_path, max_duration=25):
    """
    Split long audio files into smaller chunks for processing,
    segmenting at silence points for more natural breaks.
    """
    try:
        from pydub.silence import detect_silence

        # Check if we need to split
        audio = AudioSegment.from_file(audio_path)
        audio_duration_seconds = len(audio) / 1000

        if audio_duration_seconds <= max_duration:
            return [audio_path]

        # Detect silence in the audio (adjust parameters based on your needs)
        silence_points = detect_silence(
            audio,
            min_silence_len=500,  # Look for silences of at least 500ms
            silence_thresh=-40,  # More permissive threshold (was -35dB)
        )

        # If no silence points were detected, use time-based splitting instead
        if not silence_points:
            st.warning(
                "No silence detected in audio. Using time-based splitting instead."
            )
            segment_max_ms = max_duration * 1000
            segments = []

            # Split by fixed time chunks
            for i in range(0, len(audio), segment_max_ms):
                segment = audio[i : i + segment_max_ms]
                if len(segment) > 500:  # Only save segments longer than 500ms
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_file:
                        segment_path = temp_file.name
                    segment.export(segment_path, format="wav", parameters=["-q:a", "0"])
                    segments.append(segment_path)

            if segments:
                st.success(f"Split audio into {len(segments)} time-based segments")
                return segments
            return [audio_path]

        # Convert silence ranges to single points (end of each silence)
        silence_positions = [end for _, end in silence_points]

        # Add the end of audio as a potential split point
        silence_positions.append(len(audio))

        # Create segments based on max_duration, but snap to the next silence
        segments = []
        start_pos = 0
        segment_max_ms = max_duration * 1000

        while start_pos < len(audio):
            # Target end position based on max_duration
            target_end = start_pos + segment_max_ms

            # Find the next silence after target_end
            next_silence = target_end
            search_window = 5000  # Look up to 5 seconds past the target

            # Find the nearest silence point after target_end
            silence_found = False
            for pos in silence_positions:
                if pos > target_end:
                    # Found a silence point after target_end
                    if pos - target_end <= search_window:
                        next_silence = pos
                        silence_found = True
                    break

            # If no suitable silence found within window, use target_end directly
            if not silence_found:
                next_silence = target_end

            # Extract segment
            segment = audio[start_pos:next_silence]

            # Skip empty segments
            if len(segment) < 500:  # Skip if less than 500ms
                st.warning(f"Skipping very short segment ({len(segment)/1000:.1f}s)")
                start_pos = next_silence
                continue

            # Create a new temp file for each segment
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                segment_path = temp_file.name

            # Export to the temp file path - don't call .close() on the export result
            try:
                segment.export(segment_path, format="wav", parameters=["-q:a", "0"])

                # Verify the file exists and has content
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    # Double verify by trying to load it back
                    test_segment = AudioSegment.from_file(segment_path)
                    if len(test_segment) > 0:
                        segments.append(segment_path)
                    else:
                        st.warning(
                            f"Created segment appears empty at {start_pos/1000:.1f}s"
                        )
                        os.unlink(segment_path)
                else:
                    st.warning(
                        f"Failed to create segment at position {start_pos/1000:.1f}s"
                    )
                    if os.path.exists(segment_path):
                        os.unlink(segment_path)
            except Exception as segment_error:
                st.warning(f"Error processing segment: {str(segment_error)}")
                if os.path.exists(segment_path):
                    os.unlink(segment_path)

            # Move to the next segment
            start_pos = next_silence

            # Force garbage collection after each segment to prevent memory buildup
            gc.collect()

        # Fallback to original if segmentation failed or produced nothing useful
        if not segments:
            st.warning("No valid segments created, falling back to original audio")
            return [audio_path]

        st.success(f"Successfully split audio into {len(segments)} segments")
        return segments

    except Exception as e:
        st.error(f"Error splitting audio: {str(e)}")
        st.error("Falling back to original audio file")
        return [audio_path]  # Fall back to original file


def adjust_audio_speed(audio_path, speed_ratio):
    """
    Adjust the speed (duration) of an audio file while preserving its pitch
    using high-quality time stretching with librosa.

    Args:
        audio_path (str): Path to the input audio file.
        speed_ratio (float): Factor to adjust speed.
                             For example, if the first audio is 10 sec and you want it to match a 15 sec audio,
                             use speed_ratio = 10/15 (~0.67). A speed_ratio < 1 stretches the audio (longer duration),
                             and > 1 compresses it (shorter duration).

    Returns:
        AudioSegment: A pydub AudioSegment object with the adjusted duration and original pitch.
    """
    try:
        # Load the audio file with pydub to get info
        audio = AudioSegment.from_file(audio_path)
        sample_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width

        # Convert pydub AudioSegment to a numpy array of samples
        samples = np.array(audio.get_array_of_samples())

        # For multi-channel audio, reshape into (n_samples, channels)
        if channels > 1:
            samples = samples.reshape((-1, channels))

        # Determine maximum possible absolute value for the given sample width (assumes signed integers)
        max_val = float(2 ** (8 * sample_width - 1))

        # Normalize samples to float32 in range [-1.0, 1.0]
        samples_float = samples.astype(np.float32) / max_val

        # Use pyrubberband to time stretch the audio.
        # Note: If speed_ratio < 1, the audio is stretched (slower, longer duration);
        # if speed_ratio > 1, the audio is compressed (faster, shorter duration).
        # pyrubberband works with mono (1D) or multi-channel (2D) arrays.
        stretched = pyrb.time_stretch(samples_float, sample_rate, speed_ratio)

        # Convert the stretched audio back to integer format
        # Clip to avoid potential overflows and convert to int16 (common sample width)
        stretched_int = np.clip(stretched * max_val, -max_val, max_val - 1).astype(
            np.int16
        )

        # For multi-channel audio, flatten the array back into interleaved format
        if channels > 1:
            stretched_int = stretched_int.flatten()

        # Create a new AudioSegment from the processed raw data
        new_audio = AudioSegment(
            data=stretched_int.tobytes(),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            adjusted_path = temp_file.name
            new_audio.export(adjusted_path, format="wav")
        return adjusted_path

    except Exception as e:
        st.warning(f"Could not adjust audio with main method: {str(e)}")

        # Fall back to the simpler method if the main method fails
        try:
            # Simple fallback using only direct sample rate manipulation
            audio = AudioSegment.from_file(audio_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                adjusted_path = temp_file.name

            # Calculate a combined speed and pitch factor
            adjusted_sample_rate = int(audio.frame_rate * speed_ratio)

            # Apply the adjustment
            adjusted_audio = audio._spawn(
                audio.raw_data, overrides={"frame_rate": adjusted_sample_rate}
            )

            # Export the adjusted audio
            adjusted_audio.export(adjusted_path, format="wav")

            st.warning("Using simplified method for audio speed adjustment")
            return adjusted_path

        except Exception as e2:
            st.error(f"All audio adjustment methods failed: {str(e2)}")
            return audio_path  # Return original as last resort
