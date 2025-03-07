import os
import gc
import tempfile
from moviepy import VideoFileClip, AudioFileClip
import openai
from pydub import AudioSegment
import streamlit as st
import numpy as np
import pyrubberband as pyrb

# Additional imports for object detection and masking
import cv2
from ultralytics import YOLO, settings, models

# Initialize OpenAI client from environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load YOLO model once at module level
try:
    models.yolo.segment
    if not os.path.exists("weights"):
        os.makedirs("weights")
    model_path = "weights/yolo11x-seg.pt"
    settings.update({"weights_dir": "weights"})
    # Using YOLOv12n-seg model for instance segmentation
    yolo_model = YOLO(
        model=model_path,
        verbose=True,
    )
except Exception as e:
    yolo_model = None
    st.warning(f"Could not initialize YOLO model: {str(e)}")


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


# Functions for Person Instance Segmentation using Ultralytics YOLO SDK


def segment_image_with_yolo(image):
    """
    Process an image using Ultralytics YOLO model for person segmentation

    Args:
        image: The image as a numpy array (from OpenCV)
    Returns:
        Segmentation results from YOLO model
    """
    try:
        if yolo_model is None:
            st.warning("YOLO model not initialized")
            return None

        # Run yolov12n inference on the image
        results = yolo_model(image, conf=0.25, iou=0.45, classes=0)  # class 0 is person

        # Return the first result (should only be one since we're processing one image)
        return results[0] if results else None

    except Exception as e:
        st.warning(f"YOLO inference failed: {str(e)}")
        return None


def process_frame_with_yolo(frame):
    """
    Process a single frame with Ultralytics YOLO to detect people and replace with solid green
    """
    try:
        # Clone the frame to avoid modifying the original
        processed_frame = frame.copy()

        # Get segmentation results from YOLO
        result = segment_image_with_yolo(frame)

        if result is None or not hasattr(result, "masks") or result.masks is None:
            return frame

        # Create a combined mask for all person instances
        height, width = frame.shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Check if we have mask data
        if result.masks and len(result.masks) > 0:
            # Process each person detection mask
            for i, mask in enumerate(result.masks):
                # The masks in YOLO's output are already in the correct format
                # Convert the floating point mask to binary
                person_mask = mask.data.cpu().numpy()[0]

                # Resize mask to match frame dimensions if needed
                if person_mask.shape != (height, width):
                    person_mask = cv2.resize(person_mask, (width, height))

                # Convert to binary mask (0 or 255)
                binary_mask = (person_mask > 0.5).astype(np.uint8) * 255

                # Add to combined mask
                combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

        # Create a solid green color array
        solid_green = np.zeros_like(frame)
        solid_green[:, :] = (0, 255, 0)  # Green in BGR format

        # Convert the combined mask to 3 channels for boolean indexing
        mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])

        # Apply the solid green color directly to the masked areas
        # This completely replaces the original pixels with green color where the mask exists
        np.copyto(processed_frame, solid_green, where=(mask_3ch > 0))

        return processed_frame

    except Exception as e:
        st.warning(f"Error processing frame with YOLO: {str(e)}")
        return frame


def apply_person_segmentation_to_video(
    video_path,
    status_text=None,
    progress_bar=None,
):
    """
    Process video by applying person segmentation and green mask to each frame
    using Ultralytics YOLO SDK for accurate person masks

    Args:
        video_path: Path to the video file
        status_text: A Streamlit text element for status updates
        progress_bar: A Streamlit progress bar element
    """
    try:
        with st.spinner("Processing video with YOLO for person segmentation..."):
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                return None

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create output video file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_output:
                output_path = temp_output.name

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Process video frames
            frame_count = 0

            # Calculate frame processing interval based on video length
            # For very long videos, we might process fewer frames
            processing_interval = max(1, total_frames // 100)

            if status_text:
                status_text.text("Starting video processing with YOLO...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame_with_yolo(frame)

                # Write the processed frame
                out.write(processed_frame)

                # Update progress
                frame_count += 1
                if frame_count % processing_interval == 0:
                    progress = frame_count / total_frames
                    if progress_bar:
                        progress_bar.progress(progress)
                    if status_text:
                        status_text.text(
                            f"Processing video with YOLO: {int(progress * 100)}% complete"
                        )

            # Release resources
            cap.release()
            out.release()

            # Force garbage collection
            gc.collect()

            # Ensure the output file was created successfully
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                if status_text:
                    status_text.text("Video processing with YOLO complete!")
                return output_path
            else:
                st.error("Failed to create output video file")
                return None

    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        return None
