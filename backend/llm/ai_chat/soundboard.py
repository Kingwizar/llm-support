import sounddevice as sd
import soundfile as sf
import numpy as np

AUDIO_FILE = "none.wav"
TARGET_DEVICE = "CABLE Input"  # VB-Audio Virtual Cable

print("🎧 Liste des périphériques disponibles :\n")
devices = sd.query_devices()
for i, d in enumerate(devices):
    print(f"[{i}] {d['name']}")

# ======================================================
# Trouver le bon device VB-CABLE
# ======================================================
device_index = None
for i, dev in enumerate(devices):
    if TARGET_DEVICE.lower() in dev['name'].lower():
        device_index = i
        break

if device_index is None:
    raise RuntimeError("❌ Impossible de trouver 'CABLE Input' ! Vérifie VB-Audio Virtual Cable.")

print(f"\n✅ Périphérique trouvé : {devices[device_index]['name']} (index {device_index})")

# ======================================================
# Charger l'audio
# ======================================================
data, samplerate = sf.read(AUDIO_FILE)

# Si stéréo → convertir en mono
if len(data.shape) > 1:
    print("🔄 Conversion stéréo → mono")
    data = np.mean(data, axis=1)

print(f"🎵 Lecture du fichier : {AUDIO_FILE}")

# ======================================================
# Jouer dans VB-CABLE
# ======================================================
sd.play(data, samplerate, device=device_index)
sd.wait()

print("🟢 Audio envoyé au micro virtuel (CABLE Input).")