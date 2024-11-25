import pandas as pd
import requests
import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import io
from datetime import datetime
from transformers import pipeline
from funasr import AutoModel
import re

def download_youtube_audio(youtube_url,output_audio_path="output_audio.mp3"):

    """
    Downloads audio from a YouTube video for the given timestamp range.

    :param youtube_url: URL of the YouTube video.
    :param output_audio_path: Path to save the trimmed audio file.
    :return: Path to the saved audio file.

    """


    # Download the audio using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_audio_path,  # Save directly to the desired output path
        'postprocessor_args': [
            '-acodec', 'pcm_s16le',  # Use PCM format for .wav
            '-ar', '44100',  # Set sample rate to 44.1kHz
            '-loglevel', 'error'
        ],
        'postprocessors': [
            {'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav','preferredquality': '192'}
        ],
        'quiet': True  # Set to True to suppress yt-dlp logs
    }

    try:
        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"audio saved to: {output_audio_path}")
    except Exception as e:
        print(f"Error downloading and trimming audio: {e}")
        return None

def trimm_audio(audio_path, start_time, end_time, output_audio_path="output_audio.mp3"):

    """

    Trim audio to the given timestamp range.

    :param audio_path: Path to the input audio file.
    :param start_time: Start time in seconds.
    :param end_time: End time in seconds.
    :param output_audio_path: Path to save the trimmed audio file.
    :return: Path to the saved audio file.

    """

    # Use ffmpeg to trim the audio
    command = f"ffmpeg -i {audio_path}.wav -ss {start_time} -to {end_time} -c copy {output_audio_path}.wav"

    try:

        os.system(command)
        # Save the trimmed audio
        print(f"Trimmed audio saved to: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Error trimming audio: {e}")
        return None

class SourceFromAnnoMI:

    """
    Class to fetch video from the AnnoMI dataset with the elder therapist and client and construct conversation and audio datasets.
    """

    def __init__(self, csvurl,client_ids,therapist_ids,destination_transcript_path,destination_audio_path):

        """
        Initialize the SourceFromAnnoMI object.
        :param csvurl: URL of the AnnoMI dataset CSV file.
        :param client_ids: List of ids of transcript in which the client is identified as elder in the AnnoMI dataset .
        :param therapist_ids: List of ids of transcript in which the therapist is identified as elder in the AnnoMI dataset.
        :param destination_transcript_path: Path to save the conversation dataset.
        :param destination_audio_path: Path to save the audio

        """

        # Initialize the client and therapist IDs
        self.client_ids = client_ids
        self.therapist_ids = therapist_ids

        # Load the data from the CSV file
        self.load_data_from_file(csvurl)

        # Initialize the destination paths
        self.destination_transcript_path = destination_transcript_path

        # Initialize the destination paths
        self.destination_audio_path = destination_audio_path
        
        # Initialize the voice emotion prediction model
        model_dir = "iic/SenseVoiceSmall"

        # Load the voice emotion prediction model
        self.voice_emotion_prediction_model = AutoModel(
            model=model_dir,
            remote_model="./SenseVoice_model.py",
            trust_remote_code=True,  
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            ban_emo_unk=True,
        )

    
    def load_data_from_file(self,csvurl):

        """
        Load data from one csvfile.
        :param csvurl: URL of the AnnoMI dataset CSV file.

        """

        # Download the file
        response = requests.get(csvurl)
        if response.status_code == 200:
            print(f"File downloaded successfully ")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

        # Convert the content to a file-like object
        csv_content = io.BytesIO(response.content)

        # Load the data into a DataFrame
        self.dataframe = pd.read_csv(csv_content)

    def get_timestamps(self):

        """
        Extract the timestamps for each client and therapist in the conversation.

        """

        # Filter the DataFrame for the client and therapist IDs
        transcript_client_df = self.dataframe[self.dataframe['transcript_id'].isin(self.client_ids)]

        all_client_timestamp = {}

        # Group by transcript_id
        for transcript_id, group in transcript_client_df.groupby('transcript_id'):
            
            # Initialize a list to store the client timestamps
            client_timestamps = []
            
            # Iterate through the grouped DataFrame
            for i in range(len(group) - 1):
                # Check if the current and next row are both clients
                if group.iloc[i]['interlocutor'] == 'client':
                    start_time = group.iloc[i]['timestamp']
                    end_time = group.iloc[i + 1]['timestamp']
                    client_timestamps.append((start_time, end_time))
            
            # Store the result for this transcript
            all_client_timestamp[transcript_id] = client_timestamps
        
        # Store the client timestamps
        self.client_timestamps = all_client_timestamp

        # Filter the DataFrame for the therapist IDs
        transcript_therapist_df = self.dataframe[self.dataframe['transcript_id'].isin(self.therapist_ids)]

        # Initialize a dictionary to store all therapist timestamps
        all_therapist_timestamp = {}

        # Find all therapist timestamps for each transcript
        for transcript_id, group in transcript_therapist_df.groupby('transcript_id'):
            therapist_timestamps = []

            for i in range(len(group) - 1):
                if group.iloc[i]['interlocutor'] == 'therapist':

                    start_time = group.iloc[i]['timestamp']
                    end_time = group.iloc[i + 1]['timestamp']

                    therapist_timestamps.append((start_time, end_time))
            
            all_therapist_timestamp[transcript_id] = therapist_timestamps

        self.therapist_timestamps = all_therapist_timestamp

    def download_audio(self):

        """
        Download and trimm the audio for each specified client and therapist in the videos in Anno-MI dataset.

        """

        for therapist_id in self.therapist_ids:
           
            # Get the YouTube URL for the therapist
            youtube_url = self.dataframe[self.dataframe['transcript_id'] == therapist_id]['video_url'].values[0]
            
            # Download the audio for the therapist
            download_youtube_audio(youtube_url, output_audio_path=f"audio_therapist/{therapist_id}_therapist_audio")

            for i,(start_time,end_time) in enumerate(self.therapist_timestamps[therapist_id]):

                # Trim the audio for the therapist
                trimm_audio(f"audio_therapist/{therapist_id}_therapist_audio", start_time, end_time, output_audio_path=f"audio_therapist/{therapist_id}_therapist_audio_{i}")

        for client_id in self.client_ids:
            
            # Get the YouTube URL for the client
            youtube_url = self.dataframe[self.dataframe['transcript_id'] == client_id]['video_url'].values[0]
            
            # Download the audio for the client
            download_youtube_audio(youtube_url, output_audio_path=f"audio_client/{client_id}_client_audio")

            for i,(start_time,end_time) in enumerate(self.client_timestamps[client_id]):
                
                # Trim the audio for the client
                trimm_audio(f"audio_client/{client_id}_client_audio", start_time, end_time, output_audio_path=f"audio_client/{client_id}_client_audio_{i}")

    def get_audio(self):

        """
        Get the audio for each specified client and therapist in the videos in Anno-MI dataset.

        """

        self.get_timestamps()
        self.download_audio()

    def get_emotional_status(self,text,audio_path,type):
        
        """
        Get the emotional status of the given text or audio.
        :param text: Text to analyze.
        :param audio_path: Path to the audio file.
        :param type: Type of the input, either 'text' or 'audio'.

        """

        if type == 'text':

            # Load an emotion analysis pipeline
            RoBERTa_emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

            # Analyze the emotion of the text
            result = RoBERTa_emotion_pipeline(text)

            return [result[0]['label'],result[0]['score']]
        
        if type == 'audio':
            
            # Generate emotion prediction for the audio
            res=self.voice_emotion_prediction_model.generate(
                input=audio_path,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,  #
                merge_length_s=15,
            )

            # Get the emotional status from the audio
            matches = re.findall(r"<\|([^|]+)\|>", res[0]['text'])

            if len(matches)>=2:
                return matches[1].lower()
            else:
                return None
                        
    def construct_conversation_dataset(self):
        
        """
        Construct a conversation dataset from the AnnoMI dataset

        """

        # Initialize the conversation DataFrame
        conversation_df = pd.DataFrame(columns=['conversation_id', 'text', 'emotion','type','human_type'])

        # For all client IDs
        for client_id in self.client_ids:
            
            # Filter the DataFrame for the client
            client_df = self.dataframe[(self.dataframe['transcript_id'] == client_id) & (self.dataframe['interlocutor'] == 'client')][['transcript_id','utterance_text','interlocutor']]
            client_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id', 'utterance_text': 'text','interlocutor':'human_type'}

            client_df_renamed = client_df.rename(columns=column_mapping)

            conversation_df = pd.concat([conversation_df, client_df_renamed], ignore_index=True)

        # For all therapist IDs
        for therapist_id in self.therapist_ids:
            
            # Filter the DataFrame for the therapist
            therapist_df = self.dataframe[(self.dataframe['transcript_id'] == therapist_id) & (self.dataframe['interlocutor'] == 'therapist')][['transcript_id','utterance_text','interlocutor','video_url']]
            therapist_df['type']='interview'
            
            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id', 'utterance_text': 'text','interlocutor':'human_type'}

            therapist_df_renamed = therapist_df.rename(columns=column_mapping)

            conversation_df = pd.concat([conversation_df, therapist_df_renamed], ignore_index=True)

        # Sort the conversation DataFrame by conversation_id
        conversation_df.sort_values(by='conversation_id',inplace=True)
        
        conversation_df.groupby('conversation_id').apply(lambda x: x.reset_index(drop=True))
        
        conversation_ids = conversation_df['conversation_id'].unique().tolist()

        # Reset the conversation id
        conversation_df['conversation_id'] = conversation_df['conversation_id'].apply(lambda x: conversation_ids.index(x))

        # Get the text id for each conversation
        conversation_df['text_id'] = conversation_df.groupby('conversation_id').cumcount()

        # Set the emotional status for each text
        conversation_df['emotion_label']=conversation_df['text'].apply(lambda x: self.get_emotional_status(x,"",'text')[0])

        # Set the emotional score for each text
        conversation_df['emotion_score']=conversation_df['text'].apply(lambda x: self.get_emotional_status(x,"",'text')[1])

        # Save the conversation DataFrame to a CSV file
        conversation_df.to_csv(self.destination_transcript_path,index=False)
    
    def construct_audio_dataset(self):
        
        """
        Construct an audio dataset from the AnnoMI dataset

        """

        # Initialize the audio DataFrame
        audio_df = pd.DataFrame(columns=['conversation_id','audio_path','audio_id','emotion','type','human_type'])

        for client_id in self.client_ids:
            
            # Filter the DataFrame for the client
            client_df = self.dataframe[(self.dataframe['transcript_id'] == client_id) & (self.dataframe['interlocutor']=='client')][['transcript_id','video_url','interlocutor']]

            # Set the type of the audio
            client_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id','interlocutor':'human_type'}
        
            client_df_renamed = client_df.rename(columns=column_mapping)

            audio_df = pd.concat([audio_df,client_df_renamed],ignore_index=True)

        for therapist_id in self.therapist_ids:
            
            # Filter the DataFrame for the therapist
            therapist_df = self.dataframe[(self.dataframe['transcript_id'] == therapist_id) & (self.dataframe['interlocutor']=='therapist')][['transcript_id','video_url','interlocutor']]

            # Set the type of the audio
            therapist_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id','interlocutor':'human_type'}

            therapist_df_renamed = therapist_df.rename(columns=column_mapping)

            audio_df = pd.concat([audio_df,therapist_df_renamed],ignore_index=True)
        
        # Sort the audio DataFrame by conversation_id
        audio_df.sort_values(by='conversation_id',inplace=True)

        audio_df.groupby('conversation_id').apply(lambda x: x.reset_index(drop=True))

        audio_ids = audio_df['conversation_id'].unique().tolist()

        # Get the audio id for each conversation
        audio_df['audio_id'] = audio_df.groupby('conversation_id').cumcount()

        # Set the audio path for each audio file
        audio_df['audio_path'] = audio_df.apply(lambda x: f"audio_{x['human_type']}/{x['conversation_id']}_{x['human_type']}_audio_{x['audio_id']}.wav",axis=1)

        # Filter the audio DataFrame for existing audio files
        audio_df = audio_df[audio_df['audio_path'].apply(lambda x: os.path.exists(x))]

        # Reset the conversation id
        audio_df['conversation_id'] = audio_df['conversation_id'].apply(lambda x: audio_ids.index(x))

        # Get the audio id for each conversation
        audio_df['audio_id'] = audio_df.groupby('conversation_id').cumcount()

        # Set the emotional status for each audio file
        audio_df['emotion'] = audio_df['audio_path'].apply(lambda x: self.get_emotional_status("",x,'audio'))

        # Save the audio DataFrame to a CSV file
        audio_df.to_csv(self.destination_audio_path,index=False)                                                                            

def main():

    # Initialize the client and therapist IDs in AnnoMI dataset. 
    # The Client id is the id of transcript which client is identified as elder. 
    # The Therapist id is the id of transcript which therapist is identified as elder.
    client_ids = [0,2,8,9,10,28,31,33,36,39,56,57,60,61,65,66,76,77,79,80,91,94,97,101,113,115,121]
    therapist_ids = [14,16,18,26,28,29,34,35,37,39,46,49,51,56,60,61,64,65,77,79,82,85,93,94,97,99,121,131,133]

    # Initialize the SourceFromAnnoMI object
    source = SourceFromAnnoMI("https://raw.githubusercontent.com/uccollab/AnnoMI/refs/heads/main/AnnoMI-simple.csv",client_ids,therapist_ids,'conversation.csv','audio.csv')

    # Get the timestamps for the client and therapist
    source.get_timestamps()

    # Get the audio for the client and therapist to local
    source.get_audio()

    # Construct the conversation and audio datasets
    source.construct_conversation_dataset()
    source.construct_audio_dataset()

if __name__ == "__main__":
    main()


