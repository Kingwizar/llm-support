from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import logging, time
from llm import rag_prepare
import os
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# --- Récupération des variables d'environnement ---
APP_ENV = os.getenv("APP_ENV")
APP_PORT = int(os.getenv("APP_PORT"))
APP_HOST = os.getenv("APP_HOST")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_API_KEY = os.getenv("LLM_API_KEY")

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


# --- Connexion MongoDB ---
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]

conversations = db["conversations"]

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
                "content": m.get("content") or m.get("text") or ""
            }
            for m in conv.get("messages", [])
        ]
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

# ✅ ordre correct : content d'abord, role ensuite
class MessageCreate(BaseModel):
    content: str
    role: str

# ======================================================
# ----------------- ROUTES ------------------------------
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


# ✅ CORRIGÉ : data.content / data.role au lieu de data.text / data.isUser
@app.post("/conversations/{conv_id}/messages")
async def add_message(conv_id: str, data: MessageCreate):
    msg = {
        "role": data.role,
        "content": data.content,
        "isUser": data.role == "user"
    }
    await conversations.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"messages": msg}},
    )
    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)


@app.post("/chat", response_model=ChatResponse)
def chat(req: QuestionRequest):
    pack = rag_prepare(req.question)
    fake_summary = f"Réponse simulée pour la question: '{req.question}'"
    fake_steps = [
        "Étape 1 : Analyser la documentation associée",
        "Étape 2 : Vérifier les procédures internes",
        "Étape 3 : Contacter le support si besoin"
    ]
    citations = [
        {
            "doc": h["doc"],
            "score": h["score"],
            "snippet": next(
                (s for s in pack["sources_block"].splitlines() if h["doc"] in s),
                ""
            )[:200]
        }
        for h in pack["citations"]
    ]
    return ChatResponse(
        summary=fake_summary,
        steps=fake_steps,
        citations=citations,
        conversation_id="mock-conv-001"
    )

# ======================================================
# ----------------- MIDDLEWARE LOGGING ------------------
# ======================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = hex(id(request))
    body = await request.body()
    logger.info(f"📥 [{idem}] {request.method} {request.url} body={body.decode('utf-8')}")
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"❌ [{idem}] Exception: {str(e)}", exc_info=True)
        raise
    logger.info(f"📤 [{idem}] Done in {(time.time()-start)*1000:.2f}ms [{response.status_code}]")
    return response


# ======================================================
# ----------------- DELETE / RENAME ---------------------
# ======================================================
@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    result = await conversations.delete_one({"_id": ObjectId(conv_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    return {"success": True, "id": conv_id}


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
