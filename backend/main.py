from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
import logging, time, os
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from llm.rag_core import answer_with_rag_or_web, rag_prepare, query_ollama, query_ollama_voice_agent
from llm.prompt_builder import build_prompt_from_extracted_file
from ingest.file_ingest import extract_text_from_file
from ingest.web_search import simple_web_search
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline
import llm.agent
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time
import soundfile as sf

# Importe tes modules STT / LLM / TTS
from whisper import WhisperModel
from llm.rag_core import query_ollama_voice_agent
from kokoro import KPipeline

import shutil
from fastapi.staticfiles import StaticFiles




# ======================================================
# ----------------- CONFIGURATION ----------------------
# ======================================================

load_dotenv()

APP_ENV = os.getenv("APP_ENV")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")

# Optionnel: si tu veux forcer les URLs de download à passer par Express
# Exemple: PUBLIC_BASE_URL=http://127.0.0.1:3000/api/chat
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # p.ex. "http://127.0.0.1:3000/api/chat"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

app = FastAPI(title="LLM Chat API")
logger = logging.getLogger("uvicorn.error")

os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != [""] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# ----------------- MONGO CONNEXION --------------------
# ======================================================

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
conversations = db["conversations"]
fs = AsyncIOMotorGridFSBucket(db)

logger.info(f"🔗 MongoDB connecté à {MONGO_URI}/{MONGO_DB}")

# ======================================================
# ----------------- MODELS -----------------------------
# ======================================================

class ChatRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None

class Citation(BaseModel):
    doc: str
    score: float
    snippet: Optional[str] = ""   # <= facultatif pour éviter l’erreur de validation

class ChatResponse(BaseModel):
    summary: str
    steps: List[str]
    citations: List[Citation]
    conversation_id: str

class ConversationCreate(BaseModel):
    title: str

# ======================================================
# ----------------- HELPERS ----------------------------
# ======================================================

def clean_message(msg):
    """Nettoie les ObjectId pour être JSON-safe"""
    return {
        "_id": str(msg.get("_id")) if msg.get("_id") else None,
        "role": msg.get("role"),
        "content": msg.get("content", ""),
        "isUser": msg.get("isUser", False),
        "uploaded_at": msg.get("uploaded_at").isoformat() if msg.get("uploaded_at") else None,
        "files": [
            {
                "file_id": str(f.get("file_id")) if f.get("file_id") else None,
                "file_name": f.get("file_name"),
                "file_url": f.get("file_url"),
                "content_type": f.get("content_type"),
                "rag_context": f.get("rag_context"),
                "uploaded_at": f.get("uploaded_at").isoformat() if f.get("uploaded_at") else None,
            }
            for f in msg.get("files", [])
        ],
    }

def conv_helper(conv) -> dict:
    """Convertit un document Mongo conversation → dict sérialisable"""
    return {
        "id": str(conv["_id"]),
        "title": conv.get("title", "(Sans titre)"),
        "messages": [clean_message(m) for m in conv.get("messages", [])],
    }

def make_file_url(file_id: str) -> str:
    """Construit l’URL de téléchargement visible par le front."""
    if PUBLIC_BASE_URL:
        # p.ex. http://127.0.0.1:3000/api/chat/file/<id>
        return f"{PUBLIC_BASE_URL.rstrip('/')}/file/{file_id}"
    # sinon, lien direct FastAPI
    return f"http://127.0.0.1:{APP_PORT}/file/{file_id}"

# ======================================================
# ----------------- ROUTES CONVERSATIONS ---------------
# ======================================================

@app.get("/conversations")
async def get_conversations():
    try:
        convs = await conversations.find().to_list(100)
        return [conv_helper(c) for c in convs]
    except Exception as e:
        logger.error(f"❌ Erreur get_conversations : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    try:
        conv = await conversations.find_one({"_id": ObjectId(conv_id)})
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        return [clean_message(m) for m in conv.get("messages", [])]
    except Exception as e:
        logger.error(f"❌ Erreur get_messages : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations")
async def create_conversation(data: ConversationCreate):
    result = await conversations.insert_one({
        "title": data.title,
        "messages": [],
        "created_at": datetime.utcnow()
    })
    new_conv = await conversations.find_one({"_id": result.inserted_id})
    return conv_helper(new_conv)

@app.put("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, data: dict):
    new_title = data.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Titre manquant")
    await conversations.update_one(
        {"_id": ObjectId(conv_id)},
        {"$set": {"title": new_title}}
    )
    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    result = await conversations.delete_one({"_id": ObjectId(conv_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    return {"success": True}



# ======================================================
# ----------------- ROUTE MESSAGE ----------------------
# ======================================================

@app.post("/message/{conv_id}")
async def send_message(
    conv_id: str,
    text: str = Form(""),
    files: List[UploadFile] = File(default=[])
):
    """
    Reçoit un message utilisateur (texte + fichiers).
    Stocke chaque fichier dans GridFS, enregistre les métadonnées dans la collection 'files',
    et ajoute un unique message à la conversation.
    """
    try:
        saved_files = []

        for file in files:
            data = await file.read()

            # 1) Stockage binaire (GridFS)
            file_id = await fs.upload_from_stream(
                file.filename,
                BytesIO(data),
                metadata={
                    "content_type": file.content_type,
                    "size": len(data),
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            )

            # 2) (IMPORTANT) Métadonnées du fichier dans la collection 'files'
            await db["files"].insert_one({
                "_id": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(data),
                "uploaded_at": datetime.utcnow(),
                "conversation_id": conv_id,
            })

            # 3) Extraction + RAG (optionnelle)
            tmp_path = f"/tmp/{file.filename}"
            with open(tmp_path, "wb") as f:
                f.write(data)

            try:
                extracted = extract_text_from_file(tmp_path)
                file_prompt = build_prompt_from_extracted_file(extracted)
                rag_data = answer_with_rag_or_web(file_prompt) or {}
            except Exception as e:
                rag_data = {}
                logger.warning(f"⚠️ Extraction RAG échouée pour {file.filename}: {e}")
            finally:
                try:
                    os.remove(tmp_path)
        
                except FileNotFoundError:
                    pass

            saved_files.append({
                "file_id": str(file_id),
                "file_name": file.filename,
                "file_url": make_file_url(str(file_id)),  # <= lien vers /file/<id>
                "content_type": file.content_type,
                "rag_context": rag_data.get("sources_block", ""),
                "uploaded_at": datetime.utcnow()
            })

        # 4) Ajout d’un seul message côté conversation
        message_doc = {
            "role": "user",
            "content": (text or "").strip(),
            "isUser": True,
            "files": saved_files,
            "uploaded_at": datetime.utcnow()
        }

        await conversations.update_one(
            {"_id": ObjectId(conv_id)},
            {"$push": {"messages": message_doc}}
        )

        logger.info(f"💬 Message enregistré (texte + {len(saved_files)} fichiers) dans {conv_id}")
        return {
            "message": "✅ Enregistré",
            "files": saved_files,
            "content": (text or "").strip(),
            "conversation_id": conv_id
        }

    except Exception as e:
        logger.error(f"❌ Erreur send_message : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# ----------------- ROUTE CHAT (RAG) -------------------
# ======================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Pipeline RAG simplifié
    """
    try:
        logger.info(f"💬 Requête RAG : {req.question}")
        pack = answer_with_rag_or_web(req.question) or {}
        prompt = pack.get("prompt", "")

        ollama_answer = query_ollama(prompt, model_name="mistral-small:24b")
        # 3️⃣ Construire une réponse structurée
        summary = ollama_answer.split("\n")[0][:300] if ollama_answer else "Aucune réponse."
        steps = [line.strip() for line in ollama_answer.split("\n") if line.strip()]
        # normalise les citations pour respecter le modèle Pydantic
        citations = [
            {
                "doc": c.get("doc", ""),
                "score": float(c.get("score", 0.0)),
                "snippet": c.get("snippet", "")
            }
            for c in pack.get("citations", [])
        ]

        

        if req.conv_id:
            await conversations.update_one(
                {"_id": ObjectId(req.conv_id)},
                {"$push": {"messages": {
                    "role": "bot",
                    "content": ollama_answer,
                    "isUser": False,
                    "uploaded_at": datetime.utcnow()
                }}}
            )

        return ChatResponse(
            summary=summary,
            steps=steps,
            citations=citations,
            conversation_id=req.conv_id or "no-conv-id"
        )

    except Exception as e:
        logger.error(f"❌ Erreur /chat : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# ----------------- ROUTE FICHIERS ---------------------
# ======================================================

@app.get("/file/{file_id}")
async def get_file(file_id: str):
    try:
        oid = ObjectId(file_id)
        meta = await db["files"].find_one({"_id": oid})
        if not meta:
            logger.error(f"❌ [FastAPI] Fichier {file_id} introuvable dans 'files'")
            raise HTTPException(status_code=404, detail="Fichier introuvable")

        stream = BytesIO()
        await fs.download_to_stream(oid, stream)
        stream.seek(0)

        logger.info(f"📤 [FastAPI] Fichier envoyé : {meta['filename']} ({meta['content_type']})")
        logger.info(f"🔗 [FastAPI] URL générée : http://{APP_HOST}:{APP_PORT}/file/{file_id}")

        return StreamingResponse(
            stream,
            media_type=meta.get("content_type", "application/octet-stream"),
            headers={"Content-Disposition": f"attachment; filename={meta['filename']}"}
        )

    except Exception as e:
        logger.error(f"❌ [FastAPI] Erreur téléchargement : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur téléchargement : {str(e)}")


# ======================================================
# ----------------- WEBSEARCH --------------------------
# ======================================================

@app.post("/websearch")
async def web_search(req: dict):
    query = req.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query vide")
    results = simple_web_search(query)
    return {"query": query, "results": results}

# ======================================================
# ----------------- LOGGING GLOBAL ---------------------
# ======================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = hex(id(request))
    try:
        body = await request.body()
        body_str = body.decode("utf-8", errors="ignore")[:200]
        logger.info(f"📥 [{idem}] {request.method} {request.url} body={body_str}")
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(f"📤 [{idem}] Done in {duration:.2f}ms [{response.status_code}]")
        return response
    except Exception as e:
        logger.error(f"❌ [{idem}] Exception: {str(e)}", exc_info=True)
        raise


# ======================================================
# ----------------- ROUTE VOICE AGENT ------------------
# ======================================================

@app.post("/voice-agent")
async def voice_agent(audio: UploadFile = File(...)):
    """
    Reçoit un fichier audio, le transcrit (STT), envoie le texte au LLM
    avec un prompt vocal spécial, puis renvoie le texte + un lien audio TTS.
    """
    try:
        # 1️⃣ Sauvegarder temporairement le fichier reçu
        input_path = f"/tmp/{audio.filename}"
        with open(input_path, "wb") as f:
            f.write(await audio.read())

        # 2️⃣ Transcription audio → texte
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(input_path)
        text = " ".join([s.text for s in segments]).strip()
        print(f"🧠 Texte reconnu : {text}")

        # 3️⃣ Génération de la réponse via LLM
        from llm.rag_core import query_ollama_voice_agent
        response_text = query_ollama_voice_agent(text)

        # 4️⃣ Synthèse vocale (Kokoro)
        pipeline = KPipeline(lang_code='a')
        generator = pipeline(response_text, voice='af_sarah')

        timestamp = int(time.time())
        output_path = f"static/audio/reponse_{timestamp}.wav"

        for _, _, audio_out in generator:
            sf.write(output_path, audio_out, 24000)

        print(f"✅ Fichier TTS généré : {output_path}")

        # 5️⃣ Retourner un JSON avec lien direct
        audio_url = f"http://127.0.0.1:{APP_PORT}/{output_path}"

        return {
            "recognized_text": text,
            "response_text": response_text,
            "audio_url": audio_url
        }

    except Exception as e:
        logger.error(f"❌ Erreur /voice-agent : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Configuration
UPLOAD_DIR = "uploads"
RESPONSE_DIR = "static/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        # 1️⃣ sauvegarde du fichier reçu
        timestamp = int(time.time())
        filename = f"{timestamp}_{audio.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await audio.read())
        print(f"✅ Fichier reçu : {file_path}")

        # 2️⃣ Transcription STT
        stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = stt_model.transcribe(file_path)
        recognized_text = " ".join([s.text for s in segments]).strip()
        print(f"🧠 Texte reconnu : {recognized_text}")

        # 3️⃣ Génération réponse LLM
        response_text = query_ollama_voice_agent(recognized_text)
        print(f"🤖 Réponse LLM : {response_text}")

        # 4️⃣ Synthèse vocale TTS
        pipeline = KPipeline(lang_code='a')
        generator = pipeline(response_text, voice='af_sarah')
        output_filename = f"response_{timestamp}.wav"
        output_path = os.path.join(RESPONSE_DIR, output_filename)
        # Supposons : le pipeline renvoie des chunks audio
        with sf.SoundFile(output_path, mode='w', samplerate=24000, channels=1) as wfile:
            for _, _, audio_chunk in generator:
                wfile.write(audio_chunk)
        print(f"✅ Fichier TTS généré : {output_path}")

        # 5️⃣ Retourner JSON avec le chemin du fichier et textes
        audio_url = f"http://127.0.0.1:8000/static/audio/{output_filename}"
        return JSONResponse(content={
            "status": "success",
            "recognized_text": recognized_text,
            "response_text": response_text,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"❌ Erreur /upload-audio : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route de réponse audio (GET)
@app.get("/response-audio")
async def response_audio(filename: str):
    try:
        file_path = os.path.join(RESPONSE_DIR, filename)
        if os.path.exists(file_path):
            print(f"🎧 Téléchargement du fichier : {file_path}")
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
        else:
            return JSONResponse(content={"status": "error", "message": "Fichier introuvable"}, status_code=404)
    except Exception as e:
        print(f"❌ Erreur /response-audio : {e}")
        raise HTTPException(status_code=500, detail=str(e))