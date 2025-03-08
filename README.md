# ToSpeech

A multimedia conversion tool built with Streamlit that offers two main features:

- **PDF to Audio Conversion**: Convert PDF documents to speech with multiple voice options
- **Video Translation**: Translate video content to different languages with optional person segmentation

## Features

### PDF to Audio

- Upload and process PDF documents
- Extract text from PDF pages using OpenAI Vision
- Convert extracted text to speech with multiple voice options
- Process pages individually or in batch mode
- Merge audio files for complete document playback
- Download individual page audio or complete document audio

### Video Translation

- Upload video files in various formats (mp4, mov, avi, mkv)
- Extract audio from video
- Translate audio to your selected language using OpenAI speech models
- Optional green screen effect for people in the video using YOLO segmentation
- Merge translated audio with original video
- Download the translated video with synchronized audio

## Prerequisites

- Python 3.12 or higher
- FFmpeg
- RubberBand audio processing library
- Poppler Utils (for PDF processing)
- OpenAI API key
- Google Gemini API key (optional)

## Installation

### Option 1: Docker (Recommended)

1. Clone the repository:

   ```bash
   git clone https://your-repository-url/tospeech.git
   cd tospeech
   ```

2. Create a `.env` file in the project root with your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Build and run the Docker container:

   ```bash
   docker build -t tospeech .
   docker run -p 8501:8501 --env-file .env tospeech
   ```

4. Open your browser and navigate to `http://localhost:8501`

### Option 2: Local Installation

1. Clone the repository:

   ```bash
   git clone https://your-repository-url/tospeech.git
   cd tospeech
   ```

2. Install system dependencies:

   - For Ubuntu/Debian:
     ```bash
     sudo apt-get update && sudo apt-get install -y ffmpeg rubberband-cli poppler-utils
     ```
   - For macOS:
     ```bash
     brew install ffmpeg rubberband poppler
     ```

3. Create a Python virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root with your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

7. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Usage

### PDF to Audio

1. Navigate to the "PDF to Audio" tab in the sidebar
2. Upload a PDF file using the file uploader
3. Select a voice for the speech synthesis
4. Choose between processing individual pages or batch processing:
   - For individual pages: Select a page, click "Extract Text", then "Generate Audio"
   - For batch processing: Set the start and end page numbers, then click "Start Batch Processing"
5. Download individual page audio or the complete merged audio file

### Video Translation

1. Navigate to the "Video Translation" tab in the sidebar
2. Upload a video file using the file uploader
3. Enter the target language for translation
4. Select a voice for the speech synthesis
5. Configure advanced options if needed:
   - Enable/disable long video handling
   - Enable/disable person segmentation (green screen effect)
   - Select audio quality
6. Click "Start Translation Process"
7. Once processing is complete, you can:
   - View the translated video
   - See the original transcription and translation
   - Download the translated video

## Technical Notes

- The app uses OpenAI's latest models for text extraction, translation, and speech synthesis
- YOLO models are used for person segmentation in videos
- Large files are automatically split into manageable chunks for processing
- Audio speed is adjusted to match the original video duration

## License

[Your License Information]

## Acknowledgments

- OpenAI for providing the Vision and Speech API
- Ultralytics for YOLO object detection models
- Streamlit for the web application framework
