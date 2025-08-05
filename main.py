import sounddevice as sd ## To record and play audio.
import numpy as np
import scipy.io.wavfile as wav
import whisper ## To convert speech to text.
import cohere ## To generate smart replies to text.
from gtts import gTTS ## To convert text to speech.
import pygame ## Used only to play MP3 audio files.
import time
import os
import tempfile

co = cohere.Client("API key")  

pygame.init()

whisper_model = whisper.load_model("base")  # يمكنك تغيير "base" إلى "tiny" لتسريع الأداء

def record_audio(filename="input.wav", fs=44100, silence_threshold=10, max_duration=10): 
    
    print("تحدث الان")
    audio = [] #A list for storing recorded audio chunks.
    duration = 0 #Registration period.
    silence_duration = 0 #How long has it been without a sound?
    frame_duration = 0.5  # Every time we record only half a second.
    frames_per_chunk = int(fs * frame_duration) #Number of samples per "half a second.

    while duration < max_duration: #As long as we do not reach the maximum duration, it continues to record.
        chunk = sd.rec(frames_per_chunk, samplerate=fs, channels=2, dtype='int16')

        sd.wait()
        volume = np.abs(chunk).mean()

        if volume > silence_threshold: #If the sound is above the silence_threshold, then there is speech.
            audio.append(chunk) #We save this segment to audio.
            silence_duration = 0 #We reset the silence counter.
        else: #If there isn't enough audio:
            silence_duration += frame_duration #Increase silence_duration.
            if silence_duration > 1.5 and len(audio) > 0:  #If more than 1.5 seconds pass without audio and there is previously recorded speech, stop recording .
                break

        duration += frame_duration #We add half a second to the total recorded time.

    if audio:
        audio_data = np.concatenate(audio, axis=0) #We merge all the audio pieces into one piece using np.concatenate.
        wav.write(filename, fs, audio_data) #Save the file as WAV using wav.write.
        print("✅ تم التسجيل.")
        return True
    else:
        print(" لم يتم اكتشاف صوت.")
        return False


def transcribe_audio(audio_path):
    print("🧠 تحويل الصوت إلى نص...")
    result = whisper_model.transcribe(audio_path, language="ar") #The Whisper model is used to convert the speech in the audio file into written text in Arabic.
    return result["text"].strip() # strip():To remove extra spaces at the beginning or end.

def generate_response(user_text):
    if not user_text.strip(): #Checks if the entered text is empty or contains spaces.
        return "لم أسمع شيئًا واضحًا." #If the text is empty.
    print("🤖 توليد الرد...")
    response = co.chat(message=user_text) #The user_text is sent to the Cohere AI model using:co: The connection object to the cohere library.
    return response.text.strip()

#A function that takes text and converts it to audio.
def speak(text):
    print("🔊 تحويل النص إلى صوت...")
    tts = gTTS(text=text, lang="ar") #Convert Arabic text (lang="ar") to speech using Google Text-to-Speech library.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp: #Creates a temporary .mp3 file,delete=False: So the file is not automatically deleted after use.
        temp_path = fp.name #fp.name: Path to the temporary file.
    tts.save(temp_path) #The generated sound is saved inside the temporary file.
    pygame.mixer.music.load(temp_path) #Loads the audio file using pygame to play it.
    pygame.mixer.music.play() #Plays the audio file.
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    os.remove(temp_path) #After the run is complete, the temporary file is deleted from the device.

def main():
    if record_audio("input.wav"): #If sound is detected, the remaining steps continue.
        user_text = transcribe_audio("input.wav") #Converts the input.wav audio file to written text using the Whisper model.
        print("👤 أنت:", user_text) # Prints the text you said.
        response = generate_response(user_text) #Sends the text to Cohere, receives the response as text, and saves it in the response variable.
        print("🤖 المساعد:", response) #Converts the response to speech using gTTS.
        speak(response)

if __name__ == "__main__":
    main()