import pandas as pd
import requests
import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import io
from transformers import pipeline
from funasr import AutoModel
import re
from pyannote.audio import Pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')


class SourceFromYoutube:

    """
    Class to fetch video from YouTube and construct conversation and audio datasets.
    """

    def __init__(self,destination_transcript_path,destination_audio_path):

        """
        Initialize the SourceFromYoutube object.

        :param destination_transcript_path: Path to save the conversation dataset.
        :param destination_audio_path: Path to save the audio dataset.

        """

        # Initialize the destination paths
        self.destination_transcript_path = destination_transcript_path
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

    def download_youtube_audio(self,youtube_url,output_audio_path="output_audio.wav"):

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

    def fetch_youtube_transcript(self,video_url):

        """
        Fetch and filter captions (scripts) for a given YouTube video and timestamp.

        :param video_url: URL of the YouTube video.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :return: Transcript for the specified time range.

        """

        try:
            # Extract video ID from URL
            video_id = video_url.split("v=")[-1].split("&")[0]
            
            # Fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # Filter transcript for the desired time range
            filtered_transcript = [
                entry['text'] for entry in transcript 
    
            ]

            return "\n".join(filtered_transcript)

        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return None


    def trimm_audio(self,audio_path, start_time, end_time, output_audio_path="output_audio.mp3"):

        """
        Trim audio to the given timestamp range.

        :param audio_path: Path to the input audio file.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :param output_audio_path: Path to save the trimmed audio file.
        :return: Path to the saved audio file.

        """

        # Use ffmpeg to trim the audio
        command = f"ffmpeg -y -i {audio_path}.wav -ss {start_time} -to {end_time} -c copy {output_audio_path}.wav"

        
        try:

            os.system(command)
            # Save the trimmed audio
            print(f"Trimmed audio saved to: {output_audio_path}")
            return output_audio_path
        except Exception as e:
            print(f"Error trimming audio: {e}")
            return None
    
    def append_youtube_video(self,video_url):

        """
        Append a YouTube video to the existing conversation and audio datasets.
        :param video_url: URL of the YouTube video.

        """
        
        # Load existing CSVs
        text_df = pd.read_csv(self.destination_transcript_path)
        audio_df = pd.read_csv(self.destination_audio_path)

        # Get the last conversation_id from the existing text CSV
        last_conversation_id = text_df['conversation_id'].max() if not text_df.empty else -1
        new_conversation_id = last_conversation_id + 1
        
        # Get the last audio_id from the existing audio CSV
        last_audio_id = audio_df['conversation_id'].max() if not audio_df.empty else -1
        new_audio_id = last_audio_id + 1

        audio_path = f"audio_speaker/{new_audio_id}_speaker_audio"

        # Download the full audio
        self.download_youtube_audio(video_url,audio_path)

        # Process each diarized segment
        text_rows = []
        audio_rows = []
        audio_id = 0
        text_id = 0

        # Load the speaker diarization pipeline
        diarization_pipeline =  Pipeline.from_pretrained(
                        "pyannote/speaker-diarization",
                        use_auth_token="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"
                    )

        diarization = diarization_pipeline(f"{audio_path}.wav")

        # Fetch full transcript
        transcript_text = self.fetch_youtube_transcript(video_url)

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            # Trim audio
            output_audio_path = f"audio_speaker/{new_audio_id}_speaker_audio_{audio_id}"
            trimmed_audio_path = self.trimm_audio(audio_path, turn.start, turn.end, output_audio_path)

            print(f"{trimmed_audio_path}.wav",os.path.getsize(f"{trimmed_audio_path}.wav"))
            if os.path.getsize(f"{trimmed_audio_path}.wav") == 78:  # Check if the file size is 0 bytes
                
                os.remove(f"{trimmed_audio_path}.wav")  # Delete the file

            else:
                
                # Append audio row
                audio_rows.append({
                    "conversation_id": new_audio_id,
                    "audio_path": f"{trimmed_audio_path}.wav",
                    "audio_id": audio_id,
                    "emotion": self.get_emotional_status("", f"{trimmed_audio_path}.wav", "audio"),
                    "type": "talking",
                    "video_url": video_url
                })

            audio_id += 1
        
        
        # Merge with existing CSVs
        new_audio_df = pd.DataFrame(audio_rows)
        audio_df = pd.concat([audio_df, new_audio_df], ignore_index=True)
        audio_df.to_csv(self.destination_audio_path, index=False)

        print(f"Updated CSVs saved: {self.destination_audio_path}")    

        def sliding_window_split(text, max_length=512, stride=256):

            """
            Split text into chunks using a sliding window.
            """

            chunks = []
            for i in range(0, len(text), stride):
                chunk = text[i:i + max_length]
                chunks.append(chunk)
                if len(chunk) < max_length:
                    break
            return chunks
        
        # If exceed the maximum length, Split the transcript into sentences
        if len(transcript_text) > 512:
            
            # Split text
            chunks = sliding_window_split(transcript_text, max_length=512, stride=256)

            for chunk in chunks:
                
                # Split the chunk into sentences
                sentences = sent_tokenize(chunk)

                for sentence in sentences:

                    # Append each sentence to the text row with its emotional analysis

                    sentence = sentence.replace("\n", " ").strip()

                    text_rows.append({
                        "conversation_id": new_conversation_id,
                        "text": sentence,
                        "emotion_label": self.get_emotional_status(sentence, "", "text")[0],
                        "emotion_score": self.get_emotional_status(sentence, "", "text")[1],
                        "type": "talking",
                        "video_url": video_url,
                        "text_id": text_id
                    })
            
                    text_id += 1

        else:
            
            # Split the transcript into sentences
            sentences = sent_tokenize(transcript_text)

            for sentence in sentences:

                # Append each sentence to the text row with its emotional analysis

                sentence = sentence.replace("\n", " ").strip()

                text_rows.append({
                    "conversation_id": new_conversation_id,
                    "text": sentence,
                    "emotion_label": self.get_emotional_status(sentence, "", "text")[0],
                    "emotion_score": self.get_emotional_status(sentence, "", "text")[1],
                    "type": "talking",
                    "video_url": video_url,
                    "text_id": text_id
                })
        
                text_id += 1
        
        # Merge with existing CSVs
        new_text_df = pd.DataFrame(text_rows)
        
        # Reset the conversation id 
        text_df = pd.concat([text_df, new_text_df], ignore_index=True)
        
        
        # Save updated CSVs
        text_df.to_csv(self.destination_transcript_path, index=False)
       
        print(f"Updated CSVs saved: {self.destination_transcript_path}")
        

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

        # Initialize the SourceFromYoutube object

        self.source_from_youtube = SourceFromYoutube(destination_transcript_path,destination_audio_path)

        # Initialize the client and therapist IDs
        self.client_ids = client_ids
        self.therapist_ids = therapist_ids

        # Load the data from the CSV file
        self.load_data_from_file(csvurl)

        # Initialize the destination paths
        self.destination_transcript_path = destination_transcript_path

        # Initialize the destination paths
        self.destination_audio_path = destination_audio_path
        
    
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
            self.source_from_youtube.download_youtube_audio(youtube_url, output_audio_path=f"audio_therapist/{therapist_id}_therapist_audio")

            for i,(start_time,end_time) in enumerate(self.therapist_timestamps[therapist_id]):

                # Trim the audio for the therapist
                self.source_from_youtube.trimm_audio(f"audio_therapist/{therapist_id}_therapist_audio", start_time, end_time, output_audio_path=f"audio_therapist/{therapist_id}_therapist_audio_{i}")

        for client_id in self.client_ids:
            
            # Get the YouTube URL for the client
            youtube_url = self.dataframe[self.dataframe['transcript_id'] == client_id]['video_url'].values[0]
            
            # Download the audio for the client
            self.source_from_youtube.download_youtube_audio(youtube_url, output_audio_path=f"audio_client/{client_id}_client_audio")

            for i,(start_time,end_time) in enumerate(self.client_timestamps[client_id]):
                
                # Trim the audio for the client
                self.source_from_youtube.trimm_audio(f"audio_client/{client_id}_client_audio", start_time, end_time, output_audio_path=f"audio_client/{client_id}_client_audio_{i}")

    def get_audio(self):

        """
        Get the audio for each specified client and therapist in the videos in Anno-MI dataset.

        """

        self.get_timestamps()
        self.download_audio()

                        
    def construct_conversation_dataset(self):
        
        """
        Construct a conversation dataset from the AnnoMI dataset

        """

        # Initialize the conversation DataFrame
        self.conversation_df = pd.DataFrame(columns=['conversation_id', 'text','type','human_type'])

        # For all client IDs
        for client_id in self.client_ids:
            
            # Filter the DataFrame for the client
            client_df = self.dataframe[(self.dataframe['transcript_id'] == client_id) & (self.dataframe['interlocutor'] == 'client')][['transcript_id','utterance_text','interlocutor']]
            client_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id', 'utterance_text': 'text','interlocutor':'human_type'}

            client_df_renamed = client_df.rename(columns=column_mapping)

            self.conversation_df = pd.concat([self.conversation_df, client_df_renamed], ignore_index=True)

        # For all therapist IDs
        for therapist_id in self.therapist_ids:
            
            # Filter the DataFrame for the therapist
            therapist_df = self.dataframe[(self.dataframe['transcript_id'] == therapist_id) & (self.dataframe['interlocutor'] == 'therapist')][['transcript_id','utterance_text','interlocutor','video_url']]
            therapist_df['type']='interview'
            
            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id', 'utterance_text': 'text','interlocutor':'human_type'}

            therapist_df_renamed = therapist_df.rename(columns=column_mapping)

            self.conversation_df = pd.concat([self.conversation_df, therapist_df_renamed], ignore_index=True)

        # Sort the conversation DataFrame by conversation_id
        self.conversation_df.sort_values(by='conversation_id',inplace=True)
        
        self.conversation_df.groupby('conversation_id').apply(lambda x: x.reset_index(drop=True))
        
        conversation_ids = self.conversation_df['conversation_id'].unique().tolist()

        # Reset the conversation id
        self.conversation_df['conversation_id'] = self.conversation_df['conversation_id'].apply(lambda x: conversation_ids.index(x))

        # Get the text id for each conversation
        self.conversation_df['text_id'] = self.conversation_df.groupby('conversation_id').cumcount()

        # Set the emotional status for each text
        self.conversation_df['emotion_label']=self.conversation_df['text'].apply(lambda x: self.source_from_youtube.get_emotional_status(x,"",'text')[0])

        # Set the emotional score for each text
        self.conversation_df['emotion_score']=self.conversation_df['text'].apply(lambda x:  self.source_from_youtube.get_emotional_status(x,"",'text')[1])

        # Save the conversation DataFrame to a CSV file
        self.conversation_df.to_csv(self.destination_transcript_path,index=False)
    
    def construct_audio_dataset(self):
        
        """
        Construct an audio dataset from the AnnoMI dataset

        """

        # Initialize the audio DataFrame
        self.audio_df = pd.DataFrame(columns=['conversation_id','audio_path','audio_id','emotion','type','human_type'])

        for client_id in self.client_ids:
            
            # Filter the DataFrame for the client
            client_df = self.dataframe[(self.dataframe['transcript_id'] == client_id) & (self.dataframe['interlocutor']=='client')][['transcript_id','video_url','interlocutor']]

            # Set the type of the audio
            client_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id','interlocutor':'human_type'}
        
            client_df_renamed = client_df.rename(columns=column_mapping)

            self.audio_df = pd.concat([self.audio_df,client_df_renamed],ignore_index=True)

        for therapist_id in self.therapist_ids:
            
            # Filter the DataFrame for the therapist
            therapist_df = self.dataframe[(self.dataframe['transcript_id'] == therapist_id) & (self.dataframe['interlocutor']=='therapist')][['transcript_id','video_url','interlocutor']]

            # Set the type of the audio
            therapist_df['type']='interview'

            # Rename the columns with the target dataset column names
            column_mapping = {'transcript_id': 'conversation_id','interlocutor':'human_type'}

            therapist_df_renamed = therapist_df.rename(columns=column_mapping)

            self.audio_df = pd.concat([self.audio_df,therapist_df_renamed],ignore_index=True)
        
        # Sort the audio DataFrame by conversation_id
        self.audio_df.sort_values(by='conversation_id',inplace=True)

        self.audio_df.groupby('conversation_id').apply(lambda x: x.reset_index(drop=True))

        audio_ids = self.audio_df['conversation_id'].unique().tolist()

        # Get the audio id for each conversation
        self.audio_df['audio_id'] = self.audio_df.groupby('conversation_id').cumcount()

        # Set the audio path for each audio file
        self.audio_df['audio_path'] = self.audio_df.apply(lambda x: f"audio_{x['human_type']}/{x['conversation_id']}_{x['human_type']}_audio_{x['audio_id']}.wav",axis=1)

        # Filter the audio DataFrame for existing audio files
        self.audio_df = self.audio_df[self.audio_df['audio_path'].apply(lambda x: os.path.exists(x))]

        # Reset the conversation id
        self.audio_df['conversation_id'] = self.audio_df['conversation_id'].apply(lambda x: audio_ids.index(x))

        # Get the audio id for each conversation
        self.audio_df['audio_id'] = self.audio_df.groupby('conversation_id').cumcount()

        # Set the emotional status for each audio file
        self.audio_df['emotion'] = self.audio_df['audio_path'].apply(lambda x:  self.source_from_youtube.get_emotional_status("",x,'audio'))

        # Save the audio DataFrame to a CSV file
        self.audio_df.to_csv(self.destination_audio_path,index=False)  

def main():

    # Initialize the client and therapist IDs in AnnoMI dataset. 
    # The Client id is the id of transcript which client is identified as elder. 
    # The Therapist id is the id of transcript which therapist is identified as elder.
    client_ids = [0,2,8,9,10,28,31,33,36,39,56,57,60,61,65,66,76,77,79,80,91,94,97,101,113,115,121]
    therapist_ids = [14,16,18,26,28,29,34,35,37,39,46,49,51,56,60,61,64,65,77,79,82,85,93,94,97,99,121,131,133]

    # Initialize the SourceFromAnnoMI object
    source_from_AnnoMI = SourceFromAnnoMI("https://raw.githubusercontent.com/uccollab/AnnoMI/refs/heads/main/AnnoMI-simple.csv",client_ids,therapist_ids,'conversation.csv','audio.csv')

    # Get the timestamps for the client and therapist
    source_from_AnnoMI.get_timestamps()

    # Get the audio for the client and therapist to local
    source_from_AnnoMI.get_audio()

    # Construct the conversation and audio datasets
    source_from_AnnoMI.construct_conversation_dataset()
    source_from_AnnoMI.construct_audio_dataset()

    # Initialize the SourceFromYoutube object
    source_from_youtube = SourceFromYoutube('conversation.csv','audio.csv')

    # Append the conversation text and audio of YouTube video to the existing conversation and audio datasets
    # The parameter is the URL of the YouTube video
    source_from_youtube.append_youtube_video("https://www.youtube.com/watch?v=OcUqMQJiRz0")

if __name__ == "__main__":

    main()

    




