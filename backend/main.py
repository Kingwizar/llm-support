from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
import logging, time
from llm import rag_prepare
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
from ingest.file_ingest import extract_text_from_file

# ======================================================
# ----------------- CONFIGURATION ----------------------
# ======================================================

load_dotenv()

APP_ENV = os.getenv("APP_ENV")
APP_PORT = int(os.getenv("APP_PORT"))
APP_HOST = os.getenv("APP_HOST")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != [""] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    conv_id: str | None = None  # ⬅️ optionnel : pour insérer la réponse dans la conversation

class Citation(BaseModel):
    doc: str
    score: float
    snippet: str

class ChatResponse(BaseModel):
    summary: str
    steps: List[str]
    citations: List[Citation]
    conversation_id: str

# ======================================================
# ----------------- MONGO CONNEXION --------------------
# ======================================================

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
conversations = db["conversations"]
fs = AsyncIOMotorGridFSBucket(db)

logger.info(f"🔗 Connexion Mongo établie sur {MONGO_URI}, base = {MONGO_DB}")

# ======================================================
# ----------------- HELPERS -----------------------------
# ======================================================

def conv_helper(conv) -> dict:
    return {
        "id": str(conv["_id"]),
        "title": conv.get("title") or "(Sans titre)",
        "messages": [
            {
                "id": str(m.get("_id")) if "_id" in m else None,
                "role": m.get("role") or ("user" if m.get("isUser") else "bot"),
                "content": m.get("content") or m.get("text") or "",
            }
            for m in conv.get("messages", [])
        ],
    }

# ======================================================
# ----------------- MODELS ------------------------------
# ======================================================

class QuestionRequest(BaseModel):
    question: str

class Citation(BaseModel):
    doc: str
    score: float
    snippet: str

class ChatResponse(BaseModel):
    summary: str
    steps: List[str]
    citations: List[Citation]
    conversation_id: str

class ConversationCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    content: str
    role: str

# ======================================================
# ----------------- ROUTES CHAT -------------------------
# ======================================================

@app.get("/conversations")
async def get_conversations():
    convs = await conversations.find().to_list(100)
    return [conv_helper(c) for c in convs]

@app.post("/conversations")
async def create_conversation(data: ConversationCreate):
    result = await conversations.insert_one({
        "title": data.title,
        "messages": [],
    })
    new_conv = await conversations.find_one({"_id": result.inserted_id})
    return conv_helper(new_conv)

@app.post("/conversations/{conv_id}/messages")
async def add_message(conv_id: str, data: MessageCreate):
    msg = {
        "role": data.role,
        "content": data.content,
        "isUser": data.role == "user",
    }
    await conversations.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"messages": msg}},
    )
    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)

# ======================================================
# ----------------- ROUTE RAG / CHAT --------------------
# ======================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Appel du pipeline RAG pour générer une réponse et l’enregistrer dans la conversation.
    """
    try:
        logger.info(f"💬 Requête RAG : {req.question}")
        pack = rag_prepare(req.question)

        # 🧠 Simulation RAG
        fake_summary = f"Réponse simulée pour la question: '{req.question}'"
        fake_steps = [
            "Étape 1 : Analyser la documentation associée",
            "Étape 2 : Vérifier les procédures internes",
            "Étape 3 : Contacter le support si besoin",
        ]
        citations = [
            {
                "doc": h["doc"],
                "score": h["score"],
                "snippet": next(
                    (s for s in pack["sources_block"].splitlines() if h["doc"] in s),
                    "",
                )[:200],
            }
            for h in pack["citations"]
        ]

        # 🧩 Construction du texte de réponse
        bot_text = "\n".join(fake_steps)
        if citations:
            srcs = ", ".join([c["doc"] for c in citations])
            bot_text += f"\n📚 Sources: {srcs}"

        # 🗄️ Enregistrement dans la conversation si conv_id est fourni
        if req.conv_id:
            await conversations.update_one(
                {"_id": ObjectId(req.conv_id)},
                {"$push": {
                    "messages": {
                        "role": "bot",
                        "content": bot_text.strip(),
                        "isUser": False
                    }
                }}
            )
            logger.info(f"✅ Réponse bot enregistrée dans la conversation {req.conv_id}")
        else:
            logger.warning("⚠️ Aucun conv_id fourni, réponse non enregistrée en base")

        # ✅ Retour au frontend
        return ChatResponse(
            summary=fake_summary,
            steps=fake_steps,
            citations=citations,
            conversation_id=req.conv_id or "no-conv-id"
        )

    except Exception as e:
        logger.error(f"❌ Erreur dans /chat : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# ----------------- UPLOAD FICHIERS ---------------------
# ======================================================

@app.post("/upload/{conv_id}")
async def upload_files(conv_id: str, files: List[UploadFile] = File(...)):
    """
    Reçoit des fichiers, les stocke dans MongoDB (GridFS),
    et ajoute la référence dans la conversation.
    """
    try:
        saved_files = []
        for file in files:
            data = await file.read()
            file_id = await fs.upload_from_stream(
                file.filename,
                BytesIO(data),
                metadata={
                    "content_type": file.content_type,
                    "size": len(data),
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            )

            await db["files"].insert_one({
                "_id": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(data),
                "uploaded_at": datetime.utcnow(),
            })

            saved_files.append({
                "id": str(file_id),
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(data),
                "url": f"http://127.0.0.1:{APP_PORT}/file/{file_id}"
            })

            # 🔹 Ajoute le fichier comme message dans la conversation
            await conversations.update_one(
                {"_id": ObjectId(conv_id)},
                {"$push": {"messages": {
                    "role": "user",
                    "content": f"📎 {file.filename}",
                    "file_id": str(file_id),
                    "file_url": f"http://127.0.0.1:{APP_PORT}/file/{file_id}",
                    "isUser": True
                }}}
            )

            logger.info(f"📤 Fichier {file.filename} sauvegardé dans MongoDB GridFS.")

        return {"message": "✅ Fichiers enregistrés avec succès", "files": saved_files}

    except Exception as e:
        logger.error(f"❌ Erreur upload : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur upload : {str(e)}")

# ======================================================
# ----------------- TÉLÉCHARGEMENT FICHIERS -------------
# ======================================================

@app.get("/file/{file_id}")
async def get_file(file_id: str):
    try:
        oid = ObjectId(file_id)
        meta = await db["files"].find_one({"_id": oid})
        if not meta:
            raise HTTPException(status_code=404, detail="Fichier introuvable")

        stream = BytesIO()
        await fs.download_to_stream(oid, stream)
        stream.seek(0)

        logger.info(f"📤 Fichier téléchargé : {meta['filename']}")
        return StreamingResponse(
            stream,
            media_type=meta.get("content_type", "application/octet-stream"),
            headers={"Content-Disposition": f"attachment; filename={meta['filename']}"}
        )

    except Exception as e:
        logger.error(f"❌ Erreur téléchargement : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur téléchargement : {str(e)}")

# ======================================================
# ----------------- LOGGING GLOBAL ----------------------
# ======================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = hex(id(request))
    try:
        body = await request.body()
        body_str = body.decode("utf-8", errors="ignore")[:500]
        logger.info(f"📥 [{idem}] {request.method} {request.url} body={body_str}")
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(f"📤 [{idem}] Done in {duration:.2f}ms [{response.status_code}]")
        return response
    except Exception as e:
        logger.error(f"❌ [{idem}] Exception: {str(e)}", exc_info=True)
        raise
