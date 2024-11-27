# AIPI510-Final-Project-DataSourcing-Code

This is a code repo of datasourcing code of AIPI510-Final-Project. It includes the data sourcing code for elder person's text conversation and audio recordings and can form a automatic pipeline for constructing the conversation text and audio dataset. This dataset can be used to do better research for implementing an elder care chatbot to simulate the voice and tone of elders and talk more about elder care topics.  

## Files:
**AnnoMI-simple.csv**: The dataset containing interview videos in health settings.    
    
**audio.csv**: My audio dataset for this project.   

**conversation.csv**: My conversation dataset for this project.   

**requirements.txt**: The dependencies to run the Source_data.py correctly.     

**SenseVoice_model.py**: The external model to do emotional analysis for audio data.    

**Source_data.py**: The main sourcing data code for this final project. It fetched the videos in AnnoMI-simple.csv with elder therapists and clients and get the conversation text and segmented audio from it. To make complement, it can scrape the conversation scripts and audio data from specific videos. I scraped the conversation scripts and audio data from videos in which elders are talking about some topics related to elder care problems or health problems.    

After fetch the conversation texts and the conversation audios, I use j-hartmann/emotion-english-distilroberta-base model for text emotion recognition to get the emotion label and score of text. j-hartmann/emotion-english-distilroberta-base is an emotional pre-trained recognition model on huggingface based on DistilRoBERTa architecture, a distilled version of the robust RoBERTa language model, optimized for tasks requiring efficient inference without significant performance loss. I use pretrained SenseVoice model for audio emotion recognition to get the emotion label. SenseVoice is a comprehensive speech foundation model developed to enhance multilingual voice understanding and its functions include speech emotion recognition.

## Requirements 

In the requirements.txt file.   

## Run the Code

```pip install -r requirements.txt```

```python Source_data.py```

First the scripts can download the audio data for each client in each interview video to local folder audio_client and the audio data for each therapist in each interview video to local folder audio_therapist in AnnoMI dataset. After that the script could construct a conversation text and conversation audio dataset based on the collected audio data and the text data in AnnoMI dataset.     
      
In addition, the scripts allow user to input a specific video url on youtube of which the topic is related to elders talking about eldercare or health problems and get the segmented audio data ( Because the conversation and audio sections for full videos are too big, this code can cut it to smaller segments automatically. ) to local folder audio_speaker. After that it could add the segmented audio data and the segmented text data of this constructed text and audio dataset.      
       
Each conversation text is labeled as the emotional status and score ( to what extent the conversation is related to this emotional status) and each conversation audio is labeled as the emotional status.    

## Run the Unit Test

```pytest```

