import pytest
import os
import pandas as pd
from Source_data import SourceFromYoutube  # Replace `your_module` with your actual module name

# Test YouTube video URL (use a short, publicly accessible video)
YOUTUBE_URL = "https://www.youtube.com/watch?v=xV4pG1hjx3I"  # Replace with your test YouTube video URL

# Test output paths
TRANSCRIPT_PATH = "test_transcript.csv"
AUDIO_PATH = "test_audio.csv"


@pytest.fixture
def setup_instance():

    """
    Fixture to set up a SourceFromYoutube instance and clean up test files.

    """
    
    instance = SourceFromYoutube(TRANSCRIPT_PATH, AUDIO_PATH)

    # Ensure no old test files exist
    if os.path.exists(TRANSCRIPT_PATH):
        os.remove(TRANSCRIPT_PATH)
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)

    yield instance

    # Cleanup created test files
    if os.path.exists(TRANSCRIPT_PATH):
        os.remove(TRANSCRIPT_PATH)
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)


def test_download_youtube_audio(setup_instance):

    """
    Test downloading audio from YouTube.

    """

    instance = setup_instance
    output_audio_path = "test_audio"

    # Ensure no residual audio file exists
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    # Download the audio
    instance.download_youtube_audio(YOUTUBE_URL, output_audio_path)

    # Assert that the file was created
    assert os.path.exists(f"{output_audio_path}.wav")
    assert os.path.getsize(f"{output_audio_path}.wav") > 0

    # Cleanup
    os.remove(f"{output_audio_path}.wav")


def test_fetch_youtube_transcript(setup_instance):

    """
    Test fetching YouTube transcripts.

    """

    instance = setup_instance

    # Fetch the transcript
    transcript = instance.fetch_youtube_transcript(YOUTUBE_URL)

    # Assert transcript is not None or empty
    assert transcript is not None
    assert len(transcript.strip()) > 0


def test_trimm_audio(setup_instance):

    """
    Test trimming audio.

    """

    instance = setup_instance
    input_audio_path = "test_audio"
    trimmed_audio_path = "trimmed_audio"

    # Download test audio first
    instance.download_youtube_audio(YOUTUBE_URL, input_audio_path)

    # Ensure no residual trimmed audio file exists
    if os.path.exists(f"{trimmed_audio_path}.wav"):
        os.remove(f"{trimmed_audio_path}.wav")

    # Trim the audio (e.g., first 5 seconds)
    start_time = 0
    end_time = 5
    instance.trimm_audio(input_audio_path, start_time, end_time, trimmed_audio_path)

    # Assert that the trimmed file exists and is non-empty
    assert os.path.exists(f"{trimmed_audio_path}.wav")
    assert os.path.getsize(f"{trimmed_audio_path}.wav") > 0

    # Cleanup
    os.remove(f"{input_audio_path}.wav")
    os.remove(f"{trimmed_audio_path}.wav")

def test_get_emotional_status_text(setup_instance):

    """
    Test emotional analysis for text input.

    """

    instance = setup_instance
    text = "I am very happy today!"

    emotion_label, emotion_score = instance.get_emotional_status(text, "", "text")

    # Assert emotion results are valid
    assert emotion_label is not None
    assert isinstance(emotion_score, float)
    assert 0 <= emotion_score <= 1


def test_get_emotional_status_audio(setup_instance):

    """
    Test emotional analysis for audio input.

    """

    instance = setup_instance
    input_audio_path = "test_audio"

    # Download test audio first
    instance.download_youtube_audio(YOUTUBE_URL, input_audio_path)

    # Perform emotion analysis on the audio
    emotion = instance.get_emotional_status("", f"{input_audio_path}.wav", "audio")

    # Assert emotion results are valid
    assert emotion is not None

    # Cleanup
    os.remove(f"{input_audio_path}.wav")


