from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Depends
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
import os, time, logging
import soundfile as sf
import requests
from llm.rag_conversation import index_file_for_conversation
from io import BytesIO
from llm.rag_conversation import retrieve_from_conversation, index_file_for_conversation, extract_text


# === Modules IA ===
from faster_whisper import WhisperModel
from kokoro import KPipeline
from llm.rag_core import (
    answer_with_rag_or_web,
    query_ollama,
    query_ollama_voice_agent
)
from llm.prompt_builder import (
    build_runtime_prompt_with_memory
)

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
    allow_origins=CORS_ORIGINS,
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
# ----------------- AUTH0 CONFIG -----------------------
# ======================================================


AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "dev-5xqrzsdislhri5jj.us.auth0.com")
API_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://llm-support-api")

# cache JWKS (simple)
jwks = requests.get(f"https://{AUTH0_DOMAIN}/.well-known/jwks.json", timeout=10).json()




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
    email: Optional[EmailStr] = None
    role: str
    created_at: datetime

class ConversationCreate(BaseModel):
    title: str

class ChatRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None
    use_web: bool = False 

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
def verify_auth0_token(token: str) -> dict:
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = None
    for key in jwks.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            break

    if not rsa_key:
        raise HTTPException(status_code=401, detail="Invalid token (kid not found)")

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=API_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Auth0 token")

def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Le mot de passe est trop long (max 72 caractères)")
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)



def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# ✅ IMPORTANT : async + await Mongo
async def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    logger.info("🔐 Authorization header: %s", auth)

    if not auth or not auth.startswith("Bearer "):
        logger.warning("❌ Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ", 1)[1].strip()

    # ======================================================
    # 1️⃣ JWT LOCAL (HS256)
    # ======================================================
    try:
        logger.info("🟡 Trying LOCAL JWT (HS256)")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        logger.info("🟡 Local JWT payload: %s", payload)

        if not user_id:
            raise Exception("No sub in payload")

        user = await db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise Exception("User not found in DB")

        logger.info("✅ Authenticated LOCAL user: %s", user_id)
        return user

    except Exception as e:
        logger.info("⏭️ Not a LOCAL JWT: %s", str(e))

    # ======================================================
    # 2️⃣ AUTH0 (RS256)
    # ======================================================
    try:
        logger.info("🟡 Trying AUTH0 JWT (RS256)")
        header = jwt.get_unverified_header(token)
        logger.info("🟡 JWT header: %s", header)

        for key in jwks["keys"]:
            if key["kid"] == header["kid"]:
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=["RS256"],
                    audience=API_AUDIENCE,
                    issuer=f"https://{AUTH0_DOMAIN}/"
                )

                logger.info("✅ Authenticated AUTH0 user: %s", payload.get("sub"))

                return {
                    "_id": payload["sub"],
                    "email": payload.get("email"),
                    "username": payload.get("name") or payload.get("nickname"),
                    "role": "user",
                    "created_at": datetime.utcnow(),
                    "provider": "auth0"
                }

        raise Exception("Auth0 kid not found")

    except Exception as e:
        logger.info("⏭️ Not an AUTH0 token: %s", str(e))


    logger.error("❌ Invalid authentication token")
    raise HTTPException(status_code=401, detail="Invalid authentication token")



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
    return {"success": True, "user_id": str(result.inserted_id)}


@app.get("/debug/db-info")
async def debug_db():
    return {
        "mongo_uri": MONGO_URI,
        "mongo_db": MONGO_DB
    }




@app.post("/auth/login")
async def login(data: UserLogin):
    user = await db["users"].find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = create_access_token({
        "sub": str(user["_id"]),
        "role": user.get("role", "user")
    })

    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        id=str(user.get("_id")),
        username=user.get("username") or "Utilisateur",
        email=user.get("email") if user.get("email") else None,
        role=user.get("role", "user"),
        created_at=user.get("created_at") or datetime.utcnow()
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
        "user_id": user["_id"],  # ✅ ObjectId (mongo) ou string (auth0)
        "title": data.title.strip() or "Nouvelle conversation",
        "messages": [],
        "created_at": datetime.utcnow()
    })
    conv = await conversations.find_one({"_id": result.inserted_id})
    return conv_helper(conv)

@app.put("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, data: ConversationRename, user=Depends(get_current_user)):
    result = await conversations.update_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]},
        {"$set": {"title": data.title.strip()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user=Depends(get_current_user)):
    result = await conversations.delete_one({"_id": ObjectId(conv_id), "user_id": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    await conversation_memory.delete_one({"_id": conv_id})
    return {"success": True}

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

        extracted = extract_text(file.filename, data)

        if extracted.strip():
            index_file_for_conversation(
                conv_id=conv_id,
                filename=file.filename,
                text=extracted
            )

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
async def get_messages(conv_id: str, user=Depends(get_current_user)):
    conv = await conversations.find_one({"_id": ObjectId(conv_id), "user_id": user["_id"]})
    if not conv:
        return []
    return [clean_message(m) for m in conv.get("messages", [])]

# ======================================================
# ----------------- ROUTE CHAT (RAG) -------------------
# ======================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):

    memory = await get_memory_context(req.conv_id) if req.conv_id else ""

    # 1️⃣ RAG conversation (documents uploadés)
    raw_conv_hits = retrieve_from_conversation(req.conv_id, req.question)

    CONV_SIM_THRESHOLD = 0.45
    conv_hits = [
        h for h in raw_conv_hits
        if h.get("score", 0) >= CONV_SIM_THRESHOLD
    ]



    # 👉 Le web / RAG global ne sont utilisés QUE si aucun document utilisateur pertinent
    use_web = len(conv_hits) == 0

    # 2️⃣ RAG global + web (conditionnel)
    global_pack = answer_with_rag_or_web(req.question) if use_web else {}

    # 3️⃣ Fusion des hits (pour citations)
    all_hits = conv_hits + global_pack.get("citations", [])

    # 4️⃣ Construction hiérarchique du contexte
    sources_parts = []

    # 🔵 PRIORITÉ 1 — Documents fournis par l’utilisateur
    if conv_hits:
        sources_parts.append(
            "## 📄 Documents fournis par l’utilisateur\n" +
            "\n".join(
                f"- **{h['doc']}** : {h['text'][:500]}"
                for h in conv_hits
            )
        )

    # 🟢 PRIORITÉ 2 — RAG interne (si aucun document utilisateur)
    if not conv_hits and global_pack.get("sources_block"):
        sources_parts.append(
            "## 🧠 Base de connaissance interne N+One\n" +
            global_pack["sources_block"]
        )

    # 🟠 PRIORITÉ 3 — Web (uniquement si nécessaire)
    if use_web and global_pack.get("sources_block"):
        sources_parts.append(
            "## 🌐 Complément externe (web)\n" +
            global_pack["sources_block"]
        )

    sources_block = "\n\n".join(sources_parts)

    # 5️⃣ PROMPT FINAL (LE SEUL ENVOYÉ AU LLM)
    prompt = build_runtime_prompt_with_memory(
        question=req.question,
        hits=all_hits,
        sources_block=sources_block,
        memory_context=memory
    )

    answer = query_ollama(prompt, model_name="qwen14b_llm")

    # 6️⃣ Sauvegarde conversation + mémoire
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

        await update_conversation_memory(req.conv_id, req.question, answer)

    return ChatResponse(
        summary=answer.split("\n")[0][:300],
        steps=[l for l in answer.split("\n") if l.strip()],
        citations=all_hits,
        conversation_id=req.conv_id or "no-conv-id"
    )


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
    doc = await conversation_memory.find_one({"_id": conversation_id})
    entries = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": bot_msg}
    ]

    if doc:
        msgs = (doc["messages"] + entries)[-6:]
        await conversation_memory.update_one(
            {"_id": conversation_id},
            {"$set": {"messages": msgs, "updatedAt": datetime.utcnow()}}
        )
    else:
        await conversation_memory.insert_one({
            "_id": conversation_id,
            "messages": entries,
            "updatedAt": datetime.utcnow()
        })

async def get_memory_context(conversation_id):
    doc = await conversation_memory.find_one({"_id": conversation_id})
    if not doc:
        return ""
    return "\n".join([f"{m['role']}: {m['content']}" for m in doc["messages"]])