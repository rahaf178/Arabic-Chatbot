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


# إعداد مكتبة cohere
co = cohere.Client("636EaZPxU1EBLwUdvcxs1hxFIjqYFtOI9KEPmMgH")  # 🔑 غيّرها بمفتاحك

# إعداد pygame للصوت
pygame.init()

# إعداد Whisper
whisper_model = whisper.load_model("base")  # يمكنك تغيير "base" إلى "tiny" لتسريع الأداء

# دالة تسجيل الصوت مع إيقاف عند السكون
def record_audio(filename="input.wav", fs=44100, silence_threshold=10, max_duration=10):
     # هنا تحدد جهاز المايك (1 هو مثال، عدل الرقم حسب جهازك)
    
    print("🎤 تسجيل الصوت... (تحدث الآن)")

    audio = []
    duration = 0
    silence_duration = 0
    frame_duration = 0.5  # نصف ثانية
    frames_per_chunk = int(fs * frame_duration)

    while duration < max_duration:
        chunk = sd.rec(frames_per_chunk, samplerate=fs, channels=2, dtype='int16')

        sd.wait()
        volume = np.abs(chunk).mean()

        if volume > silence_threshold:
            audio.append(chunk)
            silence_duration = 0
        else:
            silence_duration += frame_duration
            if silence_duration > 1.5 and len(audio) > 0:  # توقف بعد 1.5 ثانية من السكون
                break

        duration += frame_duration

    if audio:
        audio_data = np.concatenate(audio, axis=0)
        wav.write(filename, fs, audio_data)
        print("✅ تم التسجيل.")
        return True
    else:
        print("🚫 لم يتم اكتشاف صوت.")
        return False


# دالة تحويل الصوت إلى نص
def transcribe_audio(audio_path):
    print("🧠 تحويل الصوت إلى نص...")
    result = whisper_model.transcribe(audio_path, language="ar")
    return result["text"].strip()

# دالة توليد الرد
def generate_response(user_text):
    if not user_text.strip():
        return "لم أسمع شيئًا واضحًا."
    print("🤖 توليد الرد...")
    response = co.chat(message=user_text)
    return response.text.strip()

# دالة تحويل النص إلى صوت
def speak(text):
    print("🔊 تحويل النص إلى صوت...")
    tts = gTTS(text=text, lang="ar")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
    tts.save(temp_path)
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    os.remove(temp_path)

# الدالة الرئيسية
def main():
    if record_audio("input.wav"):
        user_text = transcribe_audio("input.wav")
        print("👤 أنت:", user_text)
        response = generate_response(user_text)
        print("🤖 المساعد:", response)
        speak(response)

if __name__ == "__main__":
    main()