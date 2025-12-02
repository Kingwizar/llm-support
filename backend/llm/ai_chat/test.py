import sounddevice as sd
import numpy as np
from kokoro import KPipeline

TARGET_DEVICE = "CABLE Input"  # périphérique du micro virtuel


# =========================================
# Trouver le device VB-CABLE
# =========================================
devices = sd.query_devices()
device_index = None
for i, dev in enumerate(devices):
    if TARGET_DEVICE.lower() in dev['name'].lower():
        device_index = i
        break

if device_index is None:
    raise RuntimeError("❌ CABLE Input introuvable ! Vérifie VB-CABLE.")

print(f"🎧 Audio vers : {devices[device_index]['name']}")

sd.default.device = device_index


# =========================================
# Initialiser TTS Kokoro
# =========================================
pipeline = KPipeline(lang_code='a')  # français


def tts_stream(text):
    """Génère immédiatement un petit morceau audio avec Kokoro."""
    gen = pipeline(text, voice="af_sarah")
    for _, _, audio in gen:
        return np.array(audio, dtype=np.float32)


def speak(text):
    """Convertit texte → audio → micro virtuel."""
    audio = tts_stream(text)
    sd.play(audio, 24000)
    sd.wait()


# =========================================
# Mode "Tape et il parle"
# =========================================
print("\n🟢 MODE PAROLE EN DIRECT")
print("Tapes du texte : dès que tu appuies sur Entrée, la voix lit.\n")

buffer = ""

while True:
    try:
        user_input = input("👉 Tape ici : ")

        if user_input.strip() == "":
            continue

        speak(user_input)

    except KeyboardInterrupt:
        print("\n👋 Arrêt du programme.")
        break
