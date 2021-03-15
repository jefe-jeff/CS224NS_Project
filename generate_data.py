!/usr/bin/gcsfuse spoken-squad-data data

import tqdm
import os
import json


def synthesize_text(text, gci_url, filename):
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    request = {
              "audioConfig": {
                "sampleRateHertz": 16000,
                "audioEncoding": "MP3"
              },
              "input": {
                "text": text
              },
              "voice": {
                "ssmlGender": "NEUTRAL",
                "languageCode": "en-US",
                "name": "en-US-Standard-C"
              }
            }
    response = client.synthesize_speech(
        request=request
    )

    # The response's audio_content is binary.
    with open(gci_url+filename, "wb") as out:
        out.write(response.audio_content)
        
from google.cloud import speech


def transcribe_gcs_with_word_time_offsets(gcs_uri):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    result = operation.result(timeout=90)
    
    transcript_data = {}
    for result in result.results:
        alternative = result.alternatives[0]
        
        
        transcript_data["transcript"] = alternative.transcript
        transcript_data["confidence"] = alternative.confidence
        word_data = []
        start_time = []
        end_time = []
        for word_info in alternative.words:
            
            start_time += [int(word_info.start_time.total_seconds())]
            end_time += [int(word_info.end_time.total_seconds())]
        
        transcript_data["time_points"] = start_time + end_time[-1]
        
        transcript_data["sample_rate_hertz"] = 24000
        
    return transcript_data

split = 'dev'
with open('data/'+split+'-v2.0_aug.json','r') as fp:
    data = json.load(fp):

datapath = "gs://spoken-squad-data/" + split + "/"

data_list = os.listdir("data/"+split+"/")

for i, subject in enumerate(data['data']):
    print("%s/%s subjects" %(i, len(data['data'])))
    for j, paragraph in enumerate(tqdm.tqdm(subject['paragraphs'])):

        context_file = "%s_%s.flac" %(i,j)
        if not (context_file in data_list):
            context = paragraph['context']
            try:
                synthesize_text(context, "data/"+split+"/",context_file)
            except:
                pass

