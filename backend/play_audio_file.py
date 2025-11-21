# play_audio_on_micro2.py
import sounddevice as sd
import soundfile as sf
import os
import sys

# -----------------------------
# Configuration
# -----------------------------
# Index du micro que tu veux utiliser (dans la liste des "micros détectés" affichée par sd.query_devices()).
# D'après ton exemple c'était le micro d'index 2 dans la liste d'inputs.
TARGET_INPUT_INDEX_IN_INPUT_LIST = 2

# Chemin vers le fichier audio (tu peux mettre relatif ou absolu)
AUDIO_PATH_RAW = "uploads/response_audio14.wav"  # exemple ; modifie si besoin

# -----------------------------
# Résolution du chemin fichier
# -----------------------------
def resolve_audio_path(raw_path):
    # expand user
    p = os.path.expanduser(raw_path)
    # if absolute already, use it
    if os.path.isabs(p) and os.path.exists(p):
        return os.path.abspath(p)

    # try relative to cwd
    candidate = os.path.abspath(p)
    if os.path.exists(candidate):
        return candidate

    # try relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, p)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # try <script_dir>/uploads/...
    candidate = os.path.join(script_dir, "uploads", os.path.basename(p))
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # not found
    return None

audio_path = resolve_audio_path(AUDIO_PATH_RAW)
if not audio_path:
    print(f"❌ Fichier audio introuvable. J'ai essayé autour de :\n - {AUDIO_PATH_RAW}\n - cwd/{AUDIO_PATH_RAW}\n - script_dir/{AUDIO_PATH_RAW}\n - script_dir/uploads/{os.path.basename(AUDIO_PATH_RAW)}")
    sys.exit(1)

print(f"🎧 Fichier audio résolu : {audio_path}")

# -----------------------------
# Lister les périphériques
# -----------------------------
all_devices = sd.query_devices()
if not all_devices:
    print("⚠️ Aucun périphérique audio détecté (sd.query_devices() vide).")
    sys.exit(1)

# Construire la liste des périphériques d'entrée (avec leur index global)
input_devices = [(i, d) for i, d in enumerate(all_devices) if d['max_input_channels'] > 0]

if not input_devices:
    print("⚠️ Aucun périphérique d'entrée (micro) détecté.")
    sys.exit(1)

print("🎤 Micros détectés (index_global : nom) :")
for idx_global, dev in input_devices:
    print(f"  [{idx_global}] {dev['name']}  (max_in={dev['max_input_channels']}, max_out={dev['max_output_channels']})")

# Vérifier que la position TARGET_INPUT_INDEX_IN_INPUT_LIST existe
if TARGET_INPUT_INDEX_IN_INPUT_LIST < 0 or TARGET_INPUT_INDEX_IN_INPUT_LIST >= len(input_devices):
    print(f"❌ L'index ciblé {TARGET_INPUT_INDEX_IN_INPUT_LIST} est hors de la plage (0..{len(input_devices)-1}).")
    sys.exit(1)

# Récupérer le périphérique global correspondant
target_global_index, target_dev_info = input_devices[TARGET_INPUT_INDEX_IN_INPUT_LIST]
print(f"\n✅ Micro choisi (dans la liste d'inputs à la position {TARGET_INPUT_INDEX_IN_INPUT_LIST}) :")
print(f"   index_global = {target_global_index}")
print(f"   name         = {target_dev_info['name']}")
print(f"   max_input    = {target_dev_info['max_input_channels']}")
print(f"   max_output   = {target_dev_info['max_output_channels']}")

# -----------------------------
# Choix du périphérique de lecture
# -----------------------------
# Pour "jouer sur le micro", il faut que le périphérique ait des canaux de sortie (loopback/virtual cable).
# Si max_output_channels > 0 sur ce device, on l'utilise pour la lecture.
# Sinon, on essaie de trouver un périphérique de sortie qui a le même nom
playback_device_index = None

if target_dev_info['max_output_channels'] > 0:
    playback_device_index = target_global_index
else:
    # essayer de trouver un périphérique avec le même nom et qui supporte la sortie
    for i, d in enumerate(all_devices):
        if d['name'] == target_dev_info['name'] and d['max_output_channels'] > 0:
            playback_device_index = i
            break

# fallback : utiliser le périphérique de sortie par défaut
if playback_device_index is None:
    print("⚠️ Le périphérique sélectionné n'expose pas de canaux de sortie. Je vais essayer d'utiliser le périphérique de sortie par défaut.")
    try:
        default_out = sd.default.device[1]  # (in, out)
        playback_device_index = default_out
    except Exception:
        playback_device_index = None

print(f"▶️ Périphérique choisi pour lecture (index) : {playback_device_index}")

# -----------------------------
# Lecture
# -----------------------------
try:
    data, samplerate = sf.read(audio_path, always_2d=False)
except Exception as e:
    print(f"❌ Erreur lecture fichier audio : {e}")
    sys.exit(1)

# Assurer que les données sont dans un format compatible
# sounddevice accepte (N, channels) array et samplerate
try:
    if playback_device_index is not None:
        sd.default.device = (sd.default.device[0], playback_device_index)  # keep input default, set output
        print(f"ℹ️ sd.default.device défini sur: {sd.default.device} (in,out)")
    print(f"🎧 Lecture du fichier sur le périphérique (index) : {playback_device_index}")
    sd.play(data, samplerate)
    sd.wait()
    print("✅ Lecture terminée.")
except Exception as e:
    print(f"❌ Erreur pendant la lecture : {e}")
    # afficher détails utiles pour debugging
    try:
        print("Détails périphérique ciblé :", playback_device_index, "info:", all_devices[playback_device_index] if playback_device_index is not None and playback_device_index < len(all_devices) else "N/A")
    except Exception:
        pass
    sys.exit(1)
