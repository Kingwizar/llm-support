from faster_whisper import WhisperModel
from TTS.api import TTS
import torch
import os

torch.serialization.add_safe_globals([__import__("TTS.utils.radam", fromlist=["RAdam"]).RAdam])

def audio_to_text(audio_path):
    print(f"🎧 Transcription de : {audio_path}")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    text = " ".join([s.text for s in segments])
    print("🧠 Texte reconnu :")
    print(text)
    return text

def text_to_audio(text, output_path="output.wav"):
    print("🔊 Génération du fichier audio avec TTS...")
    tts = TTS("tts_models/fr/mai/tacotron2-DDC", gpu=False)
    tts.tts_to_file(text=text, file_path=output_path)
    print(f"✅ Fichier généré : {output_path}")

if __name__ == "__main__":
    audio_file = "none.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Le fichier {audio_file} n'existe pas !")
    else:
        text = audio_to_text(audio_file)
        response_text = f"Tu as dit : {text}"
        text_to_audio(response_text, "reponse.wav")
