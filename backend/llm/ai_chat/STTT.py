from datetime import datetime
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import noisereduce as nr
import wave
import scipy.signal as sg

# ==========================
# CONFIG
# ==========================
SAMPLE_RATE = 16000
FRAME_DURATION = 0.1
SILENCE_DURATION = 2
NOISE_DECAY = 0.98   # vitesse à laquelle le seuil descend
BOOST_FACTOR = 1.5   # renforce le signal après filtrage
# ==========================

print("⏳ Chargement du modèle...")
model = WhisperModel("medium", device="cuda", compute_type="int8")
print("✅ Modèle chargé\n")


def highpass_filter(audio):
    """Filtre passe-haut pour enlever basses (ventilateurs, vibrations)"""
    b, a = sg.butter(4, 100 / (SAMPLE_RATE / 2), btype='highpass')
    return sg.lfilter(b, a, audio)


def record_with_strong_vad():
    """VAD robuste pour micro bruyant."""
    frame_size = int(SAMPLE_RATE * FRAME_DURATION)
    max_silent_frames = int(SILENCE_DURATION / FRAME_DURATION)

    noise_floor = 0.02  # seuil initial pour bruits forts
    talking = False
    silent_frames = 0
    buffer = []

    print("🎧 En attente de voix...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            data, _ = stream.read(frame_size)
            frame = data[:, 0]

            # Filtre passe-haut
            frame = highpass_filter(frame)

            # Amplification
            frame = frame * BOOST_FACTOR

            # RMS sur frame
            rms = np.sqrt(np.mean(frame ** 2))

            # Adaptation du seuil en fonction du bruit
            noise_floor = max(noise_floor * NOISE_DECAY, rms * 0.3)

            if rms > noise_floor * 1.8:  # plus robuste
                if not talking:
                    print("🎤 Voix détectée, enregistrement en cours...")
                talking = True
                silent_frames = 0
                buffer.append(frame.copy())
            else:
                if talking:
                    silent_frames += 1
                    buffer.append(frame.copy())
                    if silent_frames >= max_silent_frames:
                        print("🤫 Silence détecté, arrêt.")
                        break

    if not buffer:
        return None

    return np.concatenate(buffer)


tentative = 0

while True:
    tentative += 1
    print(f"\n🔄 Tentative {tentative}")

    audio = record_with_strong_vad()

    if audio is None:
        print("⚠️ Aucun son détecté")
        continue

    # Nettoyage du bruit
    cleaned = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)

    if tentative == 3:
        filename = f"micro_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        print(f"💾 Enregistrement : {filename}")
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

    print("🧠 Transcription...")
    segments, info = model.transcribe(cleaned, language=None)
    

    for seg in segments:
        print("🗣️", seg.text.strip())

    print("-" * 40)
