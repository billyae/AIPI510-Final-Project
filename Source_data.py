import cv2
import face_recognition
from deepface import DeepFace
import pandas as pd
import requests
import os
from pytube import YouTube

def load_data_from_annoMI():
    """
    Load data from the annoMI database.
    """

    # URL to the raw file
    url = "https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-simple.csv"

    # Path to save the downloaded file
    output_file = "AnnoMI-simple.csv"

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        print(f"File downloaded successfully and saved as {output_file}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    
    return pd.read_csv(response.content)

def predict_age(frame):

    """
    Predict the age of a person in a given frame.
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    analysis = DeepFace.analyze(rgb_frame, actions=["age"], enforce_detection=False)
    return analysis["age"]

def extract_frame(video_url, timestamp):
    
    """
    Extract frame from a youtube video url and the corresponding timestamp.
    """
    
    yt = YouTube(video_url)
    stream = yt.streams.get_highest_resolution()  # Get the best quality available

    # Capture the video stream
    video_stream = stream.url
    cap = cv2.VideoCapture(video_stream)
    
    # Step 2: Calculate the frame number for the given timestamp
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_number = int(timestamp * fps)  # Frame number to extract
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Set the video position to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if not success:
        print("Error reading frame")
        return
    else:
        return frame

def client_age(video_url, timestamp):

    """
    Predict the age of a person in a given frame of a youtube video.
    """

    frame = extract_frame(video_url, timestamp)
    return predict_age(frame)

client_age("https://www.youtube.com/watch?v=PaSKcfTmFEk","0:00:13")