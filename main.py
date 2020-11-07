import os 
import tempfile

from google.cloud import texttospeech as tts
from google.cloud import storage, vision
#from __future__ import print_function

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

def process(data, context):
    file_data = data
    file_name = file_data["name"]
    bucket_name = file_data["bucket"]
    
    blob_uri = f"gs://{bucket_name}/{file_name}"

    output = find_emotion(file_name,blob_uri)
    text_to_wav('en-AU-Wavenet-A', output)


def find_emotion(fname, uri):

    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = uri
    response = client.face_detection(image=image)
    textString = ""

    print('=' * 30)
    print('File:', fname)
    for face in response.face_annotations:
        
        likelihood = vision.Likelihood(face.surprise_likelihood)
        print('Face surprised:', likelihood.name)
        textString=textString+'Face surprised, '+likelihood.name.replace("_", " ").lower()+". "
        

        likelihood = vision.Likelihood(face.anger_likelihood)
        print('Face anger:', likelihood.name)
        textString=textString+'Face anger, '+likelihood.name.replace("_", " ").lower()+". "
        

        likelihood = vision.Likelihood(face.joy_likelihood)
        print('Face joy:', likelihood.name)
        textString=textString+'Face joy, '+likelihood.name.replace("_", " ").lower()+"."
        
    return textString


def text_to_wav(voice_name, text):
    language_code = '-'.join(voice_name.split('-')[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config)

    filename = f'{language_code}.wav'
    opener = f'/tmp/{language_code}.wav'
    with open(opener, 'wb') as out:
        out.write(response.audio_content)
        #print(f'Audio content written to "{filename}"')

    upload_blob('voice-files', opener, filename)

    
    # Creating bucket object
    #bucket = client.get_bucket('voice-files')
    # Name of the object to be stored in the bucket
    #object_name_in_gcs_bucket = bucket.blob('voice-file.wav')
    # Name of the object in local file system
    #object_name_in_gcs_bucket.upload_from_filename(filename)

def upload_blob(bucket_name, source_file_name, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


    