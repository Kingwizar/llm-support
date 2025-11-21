import os
from kokoro import KPipeline
import soundfile as sf

def text_to_audio(text, output_path="nplusone_presentation.wav"):
    print("🔊 Génération du fichier audio avec Kokoro...")
    pipeline = KPipeline(lang_code='a')          # Français
    generator = pipeline(text, voice='af_sarah') # même voix que dans l’API FastAPI
    for _, _, audio in generator:
        sf.write(output_path, audio, 24000)
    print(f"✅ Fichier audio généré : {output_path}")

presentation_text = (
    "I am N+1, an advanced AI developed by Nplusone. "
   
)

if __name__ == "__main__":
    print("🎙️ Démarrage de la génération audio...")
    text_to_audio(presentation_text, "nplusone_presentation.wav")
