import os
from faster_whisper import WhisperModel
from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd
import numpy as np
import soundfile as sf
from llm.rag_core import query_ollama_voice_agent

# ==============================================
# 1️⃣ TRANSCRIPTION (audio → texte)
# ==============================================
def audio_to_text(audio_path):
    print(f"🎧 Transcription de : {audio_path}")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    text = " ".join([s.text for s in segments])
    print("🧠dgrtgqgrsdger Texte reconnu :")
    print(text)
    return text

def record_audio(output_file="input.wav", duration=5, samplerate=16000):
    print("🎙️ Enregistrement en cours... Parle maintenant !")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(output_file, audio, samplerate)
    print(f"✅ Fichier audio enregistré : {output_file}")
    return output_file

# ==============================================
# 2️⃣ SYNTHÈSE VOCALE (texte → audio, via Kokoro)
# ==============================================
def text_to_audio(text, output_path="output.wav"):
    print("🔊 Génération du fichier audio avec Kokoro...")
    pipeline = KPipeline(lang_code='en')  # f = French (français)
    generator = pipeline(text, voice='af_sarah')  # tu peux essayer d'autres voix
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        sf.write(output_path, audio, 24000)
    print(f"✅ Fichier généré : {output_path}")

# ==============================================
# 3️⃣ PIPELINE COMPLET
# ==============================================
if __name__ == "__main__":
    audio_file = "none.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Le fichier {audio_file} n'existe pas !")
    else:
        text = audio_to_text(audio_file)
        response_text = query_ollama_voice_agent(text)
        print("💬 Réponse de l'agent vocal :")
        text_to_audio(response_text, "reponsellm.wav")
