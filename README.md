# AIPI510-Final-Project-DataSourcing-Code

  This is a code repo of datasourcing code of AIPI510-Final-Project. It includes the data sourcing code for elder person's text conversation and audio recordings and can form a automatic pipeline for constructing the conversation text and audio dataset. This dataset can be used to do better research for implementing an elder care chatbot to simulate the voice and tone of elders and talk more about elder care topics.  
  
## Files:
**AnnoMI-simple.csv**: The dataset containing interview videos in health settings.    
    
**audio.csv**: My audio dataset for this project.   

**conversation.csv**: My conversation dataset for this project.   

**requirements.txt**: The dependencies to run the Source_data.py correctly.     

**SenseVoice_model.py**: The external model to do emotional analysis for audio data.    

**Source_data.py**: The main sourcing data code for this final project. It fetched the videos in AnnoMI-simple.csv with elder therapists and clients. To make complement, it can scrape the conversation scripts and audio data from specific videos. I scraped the conversation scripts and audio data from videos in which elders are talking about some topics related to elder care problems or health problems.     

## Run the Code

```pip install -r requirements.```

```python Source_data.py```

The scripts can download the audio data for each client in each interview video to local folder audio_client, the audio data for each therapist in each interview video to local folder audio_therapist and the audio data for each speaker in each videos in which elders are talking about some topics related to elder care problems or health problems to local folder audio_speaker.   

Because the conversation and audio sections for full videos are too big, this code can cut it to smaller segments automatically. Each conversation segment is labeled as the emotional status and score ( to what extent the conversation is related to this emotional status) and each audio segment is labeled as the emotional status.    
