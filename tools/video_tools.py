import os
import gc
import tempfile
from moviepy import VideoFileClip, AudioFileClip
import openai
import base64
from pydub import AudioSegment
import streamlit as st


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
            # Get video metadata
            metadata = get_video_metadata(video_path)
            if metadata:
                duration = metadata.get("format", {}).get("duration")
                if duration:
                    st.info(f"Processing video: {float(duration):.1f}s duration")

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
            # Get video metadata
            metadata = get_video_metadata(video_path)
            if metadata and metadata.get("streams"):
                video_streams = [
                    s
                    for s in metadata.get("streams", [])
                    if s.get("codec_type") == "video"
                ]
                if video_streams:
                    video_stream = video_streams[0]
                    width = video_stream.get("width", 0)
                    height = video_stream.get("height", 0)
                    fps = eval(video_stream.get("r_frame_rate", "30/1"))
                    st.info(f"Video properties: {width}x{height} @ {fps}fps")

            # Create output file path
            output_path = tempfile.mktemp(suffix=".mp4")

            # Load video and audio clips
            with VideoFileClip(video_path) as video, AudioFileClip(audio_path) as audio:

                # If the audio is shorter than the video, we need to handle that
                if audio.duration < video.duration:
                    st.warning(
                        f"The translated audio ({audio.duration:.1f}s) is shorter than the video ({video.duration:.1f}s). The remaining part will be silent."
                    )

                # Set the audio of the video
                video = video.with_audio(audio)

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

        st.info(f"Audio duration: {audio_duration_seconds:.1f}s")

        if audio_duration_seconds <= max_duration:
            return [audio_path]

        # Detect silence in the audio (adjust parameters based on your needs)
        silence_points = detect_silence(
            audio,
            min_silence_len=500,  # Look for silences of at least 500ms
            silence_thresh=-35,  # Consider anything quieter than -35dB as silence
        )

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
            for pos in silence_positions:
                if pos > target_end:
                    # Found a silence point after target_end
                    if pos - target_end <= search_window:
                        next_silence = pos
                    break

            # If no suitable silence found, cap at max additional length
            if next_silence > target_end + search_window:
                next_silence = target_end

            # Extract segment
            segment = audio[start_pos:next_silence]

            # Skip empty segments
            if len(segment) < 500:  # Skip if less than 500ms
                st.warning(f"Skipping very short segment ({len(segment)/1000:.1f}s)")
                start_pos = next_silence
                continue

            segment_path = None
            try:
                # Create a new temp file for each segment
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_file:
                    segment_path = temp_file.name

                # Export to the temp file path - explicitly close the file
                segment.export(
                    segment_path, format="wav", parameters=["-q:a", "0"]
                ).close()

                # Verify the file exists and has content
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segments.append(segment_path)
                    st.info(
                        f"Segment {len(segments)}: {len(segment)/1000:.1f}s, starts at {start_pos/1000:.1f}s"
                    )
                else:
                    st.warning(
                        f"Failed to create segment at position {start_pos/1000:.1f}s"
                    )
                    if segment_path and os.path.exists(segment_path):
                        os.unlink(segment_path)
            except Exception as segment_error:
                st.warning(f"Error processing segment: {str(segment_error)}")
                if segment_path and os.path.exists(segment_path):
                    os.unlink(segment_path)

            # Move to the next segment
            start_pos = next_silence

            # Force garbage collection after each segment to prevent memory buildup
            gc.collect()

        # Fallback to original if segmentation failed
        if not segments:
            st.warning("No valid segments created, falling back to original audio")
            return [audio_path]

        st.success(
            f"Successfully split audio into {len(segments)} segments at natural silence points"
        )
        return segments

    except Exception as e:
        st.error(f"Error splitting audio: {str(e)}")
        st.error("Falling back to original audio file")
        return [audio_path]  # Fall back to original file


def adjust_audio_speed(
    audio_path, speed_ratio, pitch_shift_semitones=None, auto_pitch_compensation=True
):
    """
    Adjust audio speed and/or pitch using high-quality processing with optional automatic pitch compensation.

    Args:
        audio_path: Path to the input audio file
        speed_ratio: Speed factor (>1 speeds up, <1 slows down)
        pitch_shift_semitones: Number of semitones to shift pitch (positive = higher, negative = lower)
                               If None and auto_pitch_compensation is True, pitch is automatically adjusted
        auto_pitch_compensation: If True and pitch_shift_semitones is None, automatically calculate
                                 pitch shift to maintain natural sound

    Returns:
        Path to the adjusted audio file
    """
    try:
        import math

        # Validate inputs to prevent domain errors
        if speed_ratio <= 0:
            st.warning(f"Invalid speed ratio ({speed_ratio}), using 1.0 instead")
            speed_ratio = 1.0

        # Limit extreme values to prevent processing issues
        if speed_ratio < 0.25:
            st.warning(f"Speed ratio too low ({speed_ratio}), limited to 0.25")
            speed_ratio = 0.25
        elif speed_ratio > 4.0:
            st.warning(f"Speed ratio too high ({speed_ratio}), limited to 4.0")

        # Create a temporary file for the adjusted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            adjusted_path = temp_file.name

        # Calculate pitch shift if automatic compensation is requested
        if pitch_shift_semitones is None and auto_pitch_compensation:
            try:
                # Formula: 12 semitones = 1 octave = doubling/halving of frequency
                # To counteract the natural pitch change from speed adjustment
                pitch_shift_semitones = -12 * math.log2(speed_ratio)
                st.info(
                    f"Auto-compensating pitch by {pitch_shift_semitones:.1f} semitones to maintain natural sound"
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pitch_shift_semitones = 0
                st.warning(
                    "Could not calculate automatic pitch compensation, using no pitch shift"
                )
        elif pitch_shift_semitones is None:
            pitch_shift_semitones = 0

        # Load audio with pydub
        audio = AudioSegment.from_file(audio_path)

        # Check if audio has content
        if len(audio) == 0:
            st.warning("Audio file is empty, returning original file")
            return audio_path

        # For pitch shifting with pydub:
        # Calculate pitch shift factor (2^(n/12) for n semitones)
        pitch_factor = 2 ** (pitch_shift_semitones / 12.0)

        # 1. First adjust pitch by changing sample rate
        original_sample_rate = audio.frame_rate
        pitch_adjusted_sample_rate = int(original_sample_rate * pitch_factor)

        # Export with the new sample rate (changes pitch and speed)
        audio.export(
            adjusted_path,
            format="wav",
            parameters=["-ar", str(pitch_adjusted_sample_rate)],
        )

        # 2. Reload with original frame rate to keep pitch change but restore duration
        adjusted_audio = AudioSegment.from_file(adjusted_path)
        adjusted_audio = adjusted_audio._spawn(
            adjusted_audio.raw_data, overrides={"frame_rate": original_sample_rate}
        )

        # 3. Now apply speed adjustment if needed
        # Calculate the speed adjustment sample rate
        speed_adjusted_sample_rate = int(original_sample_rate / speed_ratio)

        # Apply the speed change
        adjusted_audio = adjusted_audio._spawn(
            adjusted_audio.raw_data,
            overrides={"frame_rate": speed_adjusted_sample_rate},
        )

        # Export and reload to finalize changes
        adjusted_audio.export(adjusted_path, format="wav")
        adjusted_audio = AudioSegment.from_file(adjusted_path)
        adjusted_audio = adjusted_audio._spawn(
            adjusted_audio.raw_data, overrides={"frame_rate": original_sample_rate}
        )

        # Export the final version with high quality settings
        adjusted_audio.export(
            adjusted_path,
            format="wav",
            parameters=["-q:a", "0"],  # Use highest quality
        )

        st.info(
            f"Audio processed: speed={speed_ratio:.2f}x, pitch shift={pitch_shift_semitones:.1f} semitones"
        )
        return adjusted_path

    except Exception as e:
        st.warning(f"Could not adjust audio with high quality method: {str(e)}")

        # Fall back to the simpler method if the high-quality method fails
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                adjusted_path = temp_file.name

            # Use traditional PyDub approach as fallback
            original_frame_rate = audio.frame_rate
            new_frame_rate = int(
                original_frame_rate / max(speed_ratio, 0.1)
            )  # Prevent division by zero

            # Export with the new frame rate
            audio.export(
                adjusted_path, format="wav", parameters=["-ar", str(new_frame_rate)]
            )

            # Reload with original frame rate to achieve the speed change
            adjusted_audio = AudioSegment.from_file(adjusted_path)
            adjusted_audio = adjusted_audio._spawn(
                adjusted_audio.raw_data, overrides={"frame_rate": original_frame_rate}
            )

            # Export the final version with high quality settings
            adjusted_audio.export(
                adjusted_path,
                format="wav",
                parameters=["-q:a", "0"],  # Use highest quality
            )

            st.warning("Using fallback method for audio speed adjustment")
            return adjusted_path

        except Exception as e2:
            st.error(f"All audio speed and pitch adjustment methods failed: {str(e2)}")
            return audio_path
