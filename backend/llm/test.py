from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write
import numpy

print("🔄 Chargement des modèles Bark...")
preload_models()

print("🎧 Génération audio réaliste...")
audio_array = generate_audio("Bonjour, je parle avec un ton naturel, fluide et humain.", history_prompt="fr_speaker_1")

write("bark_output.wav", SAMPLE_RATE, audio_array)
print("✅ Fichier généré : bark_output.wav")
