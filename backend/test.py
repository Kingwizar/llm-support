from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, requests

app = FastAPI()

# ==========================
# ✅ Route GET simple
# ==========================
@app.get("/ping")
async def ping():
    return {"message": "✅ API en ligne et fonctionnelle !"}


# ==========================
# ✅ Route POST pour JSON Unreal
# ==========================
@app.post("/receive-data")
async def receive_data(request: Request):
    try:
        data = await request.json()
        print("📩 Données reçues depuis Unreal:", data)
        return JSONResponse(content={"status": "success", "echo": data})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


# ==========================
# ✅ Route GET avec paramètres
# ==========================
@app.get("/add")
async def add(a: int, b: int):
    return {"a": a, "b": b, "sum": a + b}


# ==========================
# ✅ Upload & réponse audio
# ==========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        # 1️⃣ Sauvegarde du fichier reçu
        filename = f"{int(time.time())}_{audio.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await audio.read())
        print(f"✅ Fichier reçu : {file_path}")

        # 2️⃣ Fichier de réponse à envoyer à Unreal
        response_audio = os.path.join(UPLOAD_DIR, "response_audio.wav")
        if not os.path.exists(response_audio):
            print("⚠️ Aucun fichier réponse trouvé.")
            return {"status": "received", "note": "Aucun fichier à renvoyer"}

        print(f"🎧 Fichier de réponse prêt : {response_audio}")
        return {"status": "success", "file_path": file_path}

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {"status": "error", "message": str(e)}


# ==========================
# ✅ Envoi du fichier audio à Unreal via téléchargement
# ==========================
@app.get("/response-audio")
async def response_audio():
    file_path = os.path.join(UPLOAD_DIR, "response_audio.wav")
    if os.path.exists(file_path):
        print("🎧 Envoi du fichier audio de réponse à Unreal")
        return FileResponse(file_path, media_type="audio/wav", filename="response_audio.wav")
    else:
        return {"status": "error", "message": "Fichier introuvable"}


# Permet d'accéder aux fichiers de /uploads depuis le navigateur
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
