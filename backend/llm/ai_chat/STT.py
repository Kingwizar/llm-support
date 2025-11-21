import speech_recognition as sr
from datetime import datetime
import noisereduce as nr
import numpy as np
import io
import wave

recognizer = sr.Recognizer()
mic = sr.Microphone()

tentative = 0

print("🎤 Parle (1 phrase). À la 3ème tentative, j'enregistre l'audio.\n")

while True:
    tentative += 1
    print(f"🔄 Tentative {tentative}")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎧 Écoute...")
        audio = recognizer.listen(source)

    print("🧠 Traitement...")

    # ---- Convertir l'audio SpeechRecognition → numpy array ----
    wav_data = audio.get_wav_data()
    wf = wave.open(io.BytesIO(wav_data), 'rb')
    raw_bytes = wf.readframes(wf.getnframes())
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    sr_audio = wf.getframerate()

    # ---- Réduction de bruit ----
    cleaned = nr.reduce_noise(y=audio_np, sr=sr_audio)

    # ---- Enregistrement seulement à la 3e tentative ----
    if tentative == 3:
        filename = f"micro_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
        print(f"💾 Audio brut enregistré : {filename}")

    # ---- Transcription Whisper ----
    try:
        text = recognizer.recognize_whisper(
            audio_data=audio,
            model="base",
            language="fr"
        )
        print("🗣️ Texte :", text)

    except Exception as e:
        print("❌ Erreur :", e)

    print("-" * 40)
