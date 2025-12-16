from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from io import BytesIO
import os, time, logging
import soundfile as sf

# === Modules IA ===
from faster_whisper import WhisperModel
from kokoro import KPipeline
from llm.rag_core import (
    answer_with_rag_or_web,
    query_ollama,
    query_ollama_voice_agent
)
from llm.prompt_builder import (
    build_prompt_from_extracted_file,
    build_runtime_prompt_with_memory
)
from ingest.file_ingest import extract_text_from_file
from ingest.web_search import simple_web_search


# ======================================================
# ----------------- CONFIGURATION ----------------------
# ======================================================
load_dotenv()

APP_ENV = os.getenv("APP_ENV")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="LLM Chat API")
logger = logging.getLogger("uvicorn.error")


# ======================================================
# ----------------- STATIC + CORS ----------------------
# ======================================================
UPLOAD_DIR = "uploads"
RESPONSE_DIR = "static/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != [""] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# ----------------- MONGO ------------------------------
# ======================================================
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
conversations = db["conversations"]
conversation_memory = db["conversation_memory"]
fs = AsyncIOMotorGridFSBucket(db)

logger.info(f"MongoDB connecté à {MONGO_URI}/{MONGO_DB}")


# ======================================================
# ----------------- MODELS -----------------------------
# ======================================================
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    username: str
    email: EmailStr
    role: str
    created_at: datetime

class ConversationCreate(BaseModel):
    title: str

class ChatRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None

class Citation(BaseModel):
    doc: Optional[str] = ""
    score: float
    snippet: Optional[str] = ""

class ChatResponse(BaseModel):
    summary: str
    steps: List[str]
    citations: List[Citation]
    conversation_id: str

class ConversationRename(BaseModel):
    title: str


# ======================================================
# ----------------- AUTH HELPERS -----------------------
# ======================================================
def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Le mot de passe est trop long (max 72 caractères)"
        )
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token invalide")

        user = await db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="Utilisateur introuvable")

        user["_id"] = str(user["_id"])
        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")


# ======================================================
# ----------------- AUTH ROUTES ------------------------
# ======================================================
@app.post("/auth/register")
async def register(user: UserRegister):
    if await db["users"].find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    doc = {
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "role": "user",
        "created_at": datetime.utcnow()
    }

    result = await db["users"].insert_one(doc)
    print(user)
    return {"success": True, "user_id": str(result.inserted_id)}

@app.post("/auth/login")
async def login(data: UserLogin):
    user = await db["users"].find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = create_access_token({
        "sub": str(user["_id"]),
        "role": user["role"]
    })

    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        id=user["_id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
        created_at=user["created_at"]
    )


# ======================================================
# ----------------- HELPERS CONVERSATION ---------------
# ======================================================
def clean_message(msg):
    return {
        "_id": str(msg.get("_id")) if msg.get("_id") else None,
        "role": msg.get("role"),
        "content": msg.get("content", ""),
        "isUser": msg.get("isUser", False),
        "uploaded_at": msg.get("uploaded_at").isoformat() if msg.get("uploaded_at") else None,
        "files": msg.get("files", [])
    }

def conv_helper(conv):
    return {
        "id": str(conv["_id"]),
        "title": conv.get("title", "(Sans titre)"),
        "messages": [clean_message(m) for m in conv.get("messages", [])],
    }

def make_file_url(file_id: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/file/{file_id}"
    return f"http://{APP_HOST}:{APP_PORT}/file/{file_id}"


# ======================================================
# ----------------- ROUTES CONVERSATIONS ---------------
# ======================================================
@app.get("/conversations")
async def get_conversations(user=Depends(get_current_user)):
    convs = await conversations.find({"user_id": user["_id"]}).to_list(100)
    return [conv_helper(c) for c in convs]

@app.post("/conversations")
async def create_conversation(data: ConversationCreate, user=Depends(get_current_user)):
    result = await conversations.insert_one({
        "user_id": user["_id"],
        "title": data.title,
        "messages": [],
        "created_at": datetime.utcnow()
    })
    conv = await conversations.find_one({"_id": result.inserted_id})
    return conv_helper(conv)


# ======================================================
# ----------------- ROUTE MESSAGE ----------------------
# ======================================================
@app.post("/message/{conv_id}")
async def send_message(
    conv_id: str,
    text: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    user=Depends(get_current_user)
):
    saved_files = []

    for file in files:
        data = await file.read()
        file_id = await fs.upload_from_stream(file.filename, BytesIO(data))
        await db["files"].insert_one({
            "_id": file_id,
            "filename": file.filename,
            "conversation_id": conv_id,
            "uploaded_at": datetime.utcnow()
        })

        saved_files.append({
            "file_id": str(file_id),
            "file_name": file.filename,
            "file_url": make_file_url(str(file_id)),
            "uploaded_at": datetime.utcnow()
        })

    message_doc = {
        "role": "user",
        "content": text.strip(),
        "isUser": True,
        "files": saved_files,
        "uploaded_at": datetime.utcnow()
    }

    await conversations.update_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]},
        {"$push": {"messages": message_doc}}
    )

    return {"success": True}

@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    try:
        conv = await conversations.find_one({"_id": ObjectId(conv_id)})

        if not conv:
            # ⚠️ IMPORTANT : conversation inexistante → messages vides
            return []

        return [clean_message(m) for m in conv.get("messages", [])]

    except Exception as e:
        logger.error(f"Erreur get_messages : {e}", exc_info=True)
        return []

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user=Depends(get_current_user)):
    result = await conversations.delete_one({
        "_id": ObjectId(conv_id),
        "user_id": user["_id"]
    })

    if result.deleted_count == 0:
        # conversation inexistante → OK logique
        return {"success": False}

    return {"success": True}


# ======================================================
# ----------------- ROUTE CHAT (RAG) -------------------
# ======================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    pack = answer_with_rag_or_web(req.question) or {}

    memory = await get_memory_context(req.conv_id) if req.conv_id else ""
    prompt = build_runtime_prompt_with_memory(
        question=req.question,
        hits=pack.get("hits", []),
        sources_block=pack.get("sources_block", ""),
        memory_context=memory
    )

    answer = query_ollama(prompt, model_name="qwen14b_llm")

    if req.conv_id:
        await conversations.update_one(
            {"_id": ObjectId(req.conv_id), "user_id": user["_id"]},
            {"$push": {"messages": {
                "role": "bot",
                "content": answer,
                "isUser": False,
                "uploaded_at": datetime.utcnow()
            }}}
        )

        await update_conversation_memory(
            req.conv_id,
            req.question,
            answer
        )

    return ChatResponse(
        summary=answer.split("\n")[0][:300],
        steps=[l for l in answer.split("\n") if l.strip()],
        citations=pack.get("citations", []),
        conversation_id=req.conv_id or "no-conv-id"
    )

@app.put("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str,
    data: ConversationRename,
    user=Depends(get_current_user)
):
    result = await conversations.update_one(
        {
            "_id": ObjectId(conv_id),
            "user_id": user["_id"]
        },
        {
            "$set": {
                "title": data.title.strip()
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)

@app.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user=Depends(get_current_user)
):
    result = await conversations.delete_one({
        "_id": ObjectId(conv_id),
        "user_id": user["_id"]
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    await db.conversation_memory.delete_one({"_id": conv_id})

    return {"success": True}


# ======================================================
# ----------------- ROUTES VOICE -----------------------
# ======================================================
@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    timestamp = int(time.time())
    file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{audio.filename}")
    with open(file_path, "wb") as f:
        f.write(await audio.read())

    stt_model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = stt_model.transcribe(file_path)
    recognized_text = " ".join([s.text for s in segments]).strip()

    response_text = query_ollama_voice_agent(recognized_text)

    pipeline = KPipeline(lang_code='a')
    generator = pipeline(response_text, voice='af_sarah')

    output_filename = f"response_{timestamp}.wav"
    output_path = os.path.join(RESPONSE_DIR, output_filename)

    with sf.SoundFile(output_path, mode='w', samplerate=24000, channels=1) as wfile:
        for _, _, chunk in generator:
            wfile.write(chunk)

    return {
        "recognized_text": recognized_text,
        "response_text": response_text,
        "audio_url": f"http://{APP_HOST}:{APP_PORT}/static/audio/{output_filename}"
    }


# ======================================================
# ----------------- MEMORY -----------------------------
# ======================================================
async def update_conversation_memory(conversation_id, user_msg, bot_msg):
    doc = await db.conversation_memory.find_one({"_id": conversation_id})
    entries = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": bot_msg}
    ]

    if doc:
        msgs = (doc["messages"] + entries)[-6:]
        await db.conversation_memory.update_one(
            {"_id": conversation_id},
            {"$set": {"messages": msgs, "updatedAt": datetime.utcnow()}}
        )
    else:
        await db.conversation_memory.insert_one({
            "_id": conversation_id,
            "messages": entries,
            "updatedAt": datetime.utcnow()
        })

async def get_memory_context(conversation_id):
    doc = await db.conversation_memory.find_one({"_id": conversation_id})
    if not doc:
        return ""
    return "\n".join([f"{m['role']}: {m['content']}" for m in doc["messages"]])
