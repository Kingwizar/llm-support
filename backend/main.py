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
from prometheus_fastapi_instrumentator import Instrumentator



# ======================================================
# ----------------- CONFIGURATION ----------------------
# ======================================================
load_dotenv()

# Application runtime environment (development / production)
APP_ENV = os.getenv("APP_ENV")

# FastAPI listening port
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# Host used mainly for URL generation
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")

# MongoDB connection URI
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB database name
MONGO_DB = os.getenv("MONGO_DB")

# Allowed origins for CORS (frontend, mobile clients, etc.)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

# Public application URL (reverse proxy, ngrok, domain)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

# Secret key for local JWT signing
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME")

# JWT signing algorithm
JWT_ALGORITHM = "HS256"

# JWT expiration time (in hours)
JWT_EXPIRE_HOURS = 24

# Password hashing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 Bearer scheme (used for docs and dependency injection)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# FastAPI application instance
app = FastAPI(title="LLM Chat API")

# Main Uvicorn logger
logger = logging.getLogger("uvicorn.error")

# Prometheus metrics exposure (/metrics)
Instrumentator().instrument(app).expose(app)


# ======================================================
# ----------------- STATIC + CORS ----------------------
# ======================================================

# Directory for uploaded user files (documents, audio, etc.)
UPLOAD_DIR = "uploads"

# Directory for generated audio responses (TTS)
RESPONSE_DIR = "static/audio"

# Ensure required directories exist at startup
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

# Mount static files directory (audio, assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS configuration for frontend and external clients
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

# Asynchronous MongoDB client
client = AsyncIOMotorClient(MONGO_URI)

# Main MongoDB database
db = client[MONGO_DB]

# Collection storing user conversations
conversations = db["conversations"]

# Collection storing summarized conversation memory
conversation_memory = db["conversation_memory"]

# GridFS bucket for binary file storage
fs = AsyncIOMotorGridFSBucket(db)

# MongoDB connection log
logger.info(f"MongoDB connected to {MONGO_URI}/{MONGO_DB}")


# ======================================================
# ----------------- AUTH0 CONFIG -----------------------
# ======================================================

# Auth0 tenant domain
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "dev-5xqrzsdislhri5jj.us.auth0.com")

# Expected audience for Auth0 access tokens
API_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://llm-support-api")

# Fetch Auth0 JWKS for RS256 token verification
try:
    jwks = requests.get(
        f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
        timeout=10
    ).json()
except Exception as e:
    logger.error("JWKS fetch failed: %s", e)
    jwks = {"keys": []}





# ======================================================
# ----------------- MODELS -----------------------------
# ======================================================

# Payload for user registration
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

# Payload for user login
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Public user representation
class UserOut(BaseModel):
    id: str
    username: str
    email: Optional[EmailStr] = None
    role: str
    created_at: datetime

# Payload to create a new conversation
class ConversationCreate(BaseModel):
    title: str

# Chat request payload
class ChatRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None
    use_web: bool = False 

# Citation structure for RAG sources
class Citation(BaseModel):
    doc: Optional[str] = ""
    score: float
    snippet: Optional[str] = ""

# Chat response returned to the client
class ChatResponse(BaseModel):
    summary: str
    steps: List[str]
    citations: List[Citation]
    conversation_id: str

# Payload to rename a conversation
class ConversationRename(BaseModel):
    title: str


# ======================================================
# ----------------- AUTH HELPERS -----------------------
# ======================================================

# Verify Auth0 RS256 access token using JWKS
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


# Hash a user password using bcrypt
def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password too long (max 72 characters)")
    return pwd_context.hash(password)

# Verify a plain password against its hash
def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


# Create a local JWT (HS256)
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# Resolve the currently authenticated user (local JWT or Auth0)
async def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    logger.info("Authorization header: %s", auth)

    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ", 1)[1].strip()

    # ======================================================
    # LOCAL JWT (HS256)
    # ======================================================
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")

        if not user_id:
            raise Exception("Missing sub claim")

        user = await db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise Exception("User not found")

        return user

    except Exception:
        pass

    # ======================================================
    # AUTH0 JWT (RS256)
    # ======================================================
    try:
        header = jwt.get_unverified_header(token)

        for key in jwks["keys"]:
            if key["kid"] == header["kid"]:
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=["RS256"],
                    audience=API_AUDIENCE,
                    issuer=f"https://{AUTH0_DOMAIN}/"
                )

                return {
                    "_id": payload["sub"],
                    "email": payload.get("email"),
                    "username": payload.get("name") or payload.get("nickname"),
                    "role": "user",
                    "created_at": datetime.utcnow(),
                    "provider": "auth0"
                }

        raise Exception("Auth0 kid not found")

    except Exception:
        pass

    raise HTTPException(status_code=401, detail="Invalid authentication token")


# ======================================================
# ----------------- AUTH ROUTES ------------------------
# ======================================================

# Register a new local user account
@app.post("/auth/register")
async def register(user: UserRegister):
    # Check if the email is already registered
    if await db["users"].find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already in use")

    # User document stored in MongoDB
    doc = {
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "role": "user",
        "created_at": datetime.utcnow()
    }

    # Insert user and return its id
    result = await db["users"].insert_one(doc)
    return {"success": True, "user_id": str(result.inserted_id)}


# Debug endpoint to validate MongoDB configuration
@app.get("/debug/db-info")
async def debug_db():
    return {
        "mongo_uri": MONGO_URI,
        "mongo_db": MONGO_DB
    }


# Authenticate user and issue a local JWT
@app.post("/auth/login")
async def login(data: UserLogin):
    # Retrieve user by email
    user = await db["users"].find_one({"email": data.email})

    # Validate credentials
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create signed JWT containing user id and role
    token = create_access_token({
        "sub": str(user["_id"]),
        "role": user.get("role", "user")
    })

    return {"access_token": token, "token_type": "bearer"}


# Return information about the currently authenticated user
@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        id=str(user.get("_id")),
        username=user.get("username") or "User",
        email=user.get("email") if user.get("email") else None,
        role=user.get("role", "user"),
        created_at=user.get("created_at") or datetime.utcnow()
    )


# ======================================================
# ----------------- HELPERS CONVERSATION ---------------
# ======================================================

# Normalize a message document for API responses
def clean_message(msg):
    return {
        "_id": str(msg.get("_id")) if msg.get("_id") else None,
        "role": msg.get("role"),
        "content": msg.get("content", ""),
        "isUser": msg.get("isUser", False),
        "uploaded_at": msg.get("uploaded_at").isoformat() if msg.get("uploaded_at") else None,
        "files": msg.get("files", [])
    }

# Normalize a conversation document with its messages
def conv_helper(conv):
    return {
        "id": str(conv["_id"]),
        "title": conv.get("title", "(Untitled)"),
        "messages": [clean_message(m) for m in conv.get("messages", [])],
    }

# Build a public URL for an uploaded file
def make_file_url(file_id: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/file/{file_id}"
    return f"http://{APP_HOST}:{APP_PORT}/file/{file_id}"


# ======================================================
# ----------------- ROUTES CONVERSATIONS ---------------
# ======================================================

# List all conversations belonging to the current user
@app.get("/conversations")
async def get_conversations(user=Depends(get_current_user)):
    convs = await conversations.find({"user_id": user["_id"]}).to_list(100)
    return [conv_helper(c) for c in convs]


# Create a new conversation
@app.post("/conversations")
async def create_conversation(data: ConversationCreate, user=Depends(get_current_user)):
    result = await conversations.insert_one({
        "user_id": user["_id"],
        "title": data.title.strip() or "New conversation",
        "messages": [],
        "created_at": datetime.utcnow()
    })

    conv = await conversations.find_one({"_id": result.inserted_id})
    return conv_helper(conv)


# Rename an existing conversation
@app.put("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str,
    data: ConversationRename,
    user=Depends(get_current_user)
):
    result = await conversations.update_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]},
        {"$set": {"title": data.title.strip()}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = await conversations.find_one({"_id": ObjectId(conv_id)})
    return conv_helper(conv)


# Delete a conversation and its associated memory
@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user=Depends(get_current_user)):
    result = await conversations.delete_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]}
    )

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Remove summarized memory for this conversation
    await conversation_memory.delete_one({"_id": conv_id})
    return {"success": True}


# ======================================================
# ----------------- ROUTE MESSAGE ----------------------
# ======================================================

# Add a user message (text + optional files) to a conversation
@app.post("/message/{conv_id}")
async def send_message(
    conv_id: str,
    text: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    user=Depends(get_current_user)
):
    # Store metadata of uploaded files linked to the message
    saved_files = []

    # Process each uploaded file
    for file in files:
        # Read file content into memory
        data = await file.read()

        # Store raw file in MongoDB GridFS
        file_id = await fs.upload_from_stream(file.filename, BytesIO(data))

        # Extract text content for RAG indexing (PDF, DOCX, etc.)
        extracted = extract_text(file.filename, data)

        # Index extracted text at conversation level if non-empty
        if extracted.strip():
            index_file_for_conversation(
                conv_id=conv_id,
                filename=file.filename,
                text=extracted
            )

        # Persist file metadata in MongoDB
        await db["files"].insert_one({
            "_id": file_id,
            "filename": file.filename,
            "conversation_id": conv_id,
            "uploaded_at": datetime.utcnow()
        })

        # Build file reference returned to the frontend
        saved_files.append({
            "file_id": str(file_id),
            "file_name": file.filename,
            "file_url": make_file_url(str(file_id)),
            "uploaded_at": datetime.utcnow()
        })

    # User message document
    message_doc = {
        "role": "user",
        "content": text.strip(),
        "isUser": True,
        "files": saved_files,
        "uploaded_at": datetime.utcnow()
    }

    # Append message to the conversation
    await conversations.update_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]},
        {"$push": {"messages": message_doc}}
    )

    return {"success": True}


# Retrieve all messages from a conversation
@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, user=Depends(get_current_user)):
    conv = await conversations.find_one(
        {"_id": ObjectId(conv_id), "user_id": user["_id"]}
    )
    if not conv:
        return []
    return [clean_message(m) for m in conv.get("messages", [])]


# ======================================================
# ----------------- ROUTE CHAT (RAG) -------------------
# ======================================================

# Main chat endpoint using conversation-level RAG + optional global/web RAG
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):

    # Retrieve short-term summarized memory for the conversation
    memory = await get_memory_context(req.conv_id) if req.conv_id else ""

    # 1. Conversation-level RAG (user uploaded documents)
    raw_conv_hits = retrieve_from_conversation(req.conv_id, req.question)

    # Similarity threshold to filter weak matches
    CONV_SIM_THRESHOLD = 0.45
    conv_hits = [
        h for h in raw_conv_hits
        if h.get("score", 0) >= CONV_SIM_THRESHOLD
    ]

    # Global RAG / web is used only if explicitly requested
    use_web = req.use_web is True

    # 2. Global knowledge base + web (conditional)
    global_pack = answer_with_rag_or_web(req.question) if use_web else {}

    # 3. Merge all hits for citation tracking
    all_hits = conv_hits + global_pack.get("citations", [])

    # 4. Hierarchical context construction
    sources_parts = []

    # Priority 1 — User-provided documents
    if conv_hits:
        sources_parts.append(
            "## User uploaded documents\n" +
            "\n".join(
                f"- **{h['doc']}** : {h['text'][:500]}"
                for h in conv_hits
            )
        )

    # Priority 2 — Internal knowledge base
    if not conv_hits and global_pack.get("sources_block"):
        sources_parts.append(
            "## Internal knowledge base\n" +
            global_pack["sources_block"]
        )

    # Priority 3 — External web sources
    if use_web and global_pack.get("sources_block"):
        sources_parts.append(
            "## External web sources\n" +
            global_pack["sources_block"]
        )

    # Final aggregated sources block
    sources_block = "\n\n".join(sources_parts)

    # 5. Final runtime prompt sent to the LLM
    prompt = build_runtime_prompt_with_memory(
        question=req.question,
        hits=all_hits,
        sources_block=sources_block,
        memory_context=memory
    )

    # Query the LLM
    answer = query_ollama(prompt, model_name="qwen14b_llm")

    # 6. Persist assistant answer and update conversation memory
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

    # Structured response returned to the client
    return ChatResponse(
        summary=answer.split("\n")[0][:300],
        steps=[l for l in answer.split("\n") if l.strip()],
        citations=all_hits,
        conversation_id=req.conv_id or "no-conv-id"
    )

# ======================================================
# ----------------- ROUTES VOICE -----------------------
# ======================================================

# Upload an audio file and return a spoken AI response
@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    # Generate a unique filename using a timestamp
    timestamp = int(time.time())
    file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{audio.filename}")

    # Persist uploaded audio file to disk
    with open(file_path, "wb") as f:
        f.write(await audio.read())

    # Speech-to-text using Whisper (audio → text)
    stt_model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = stt_model.transcribe(file_path)
    recognized_text = " ".join([s.text for s in segments]).strip()

    # Query the LLM optimized for voice interaction
    response_text = query_ollama_voice_agent(recognized_text)

    # Text-to-speech pipeline initialization
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(response_text, voice='af_sarah')

    # Output audio file path
    output_filename = f"response_{timestamp}.wav"
    output_path = os.path.join(RESPONSE_DIR, output_filename)

    # Stream generated audio chunks into a WAV file
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

# Update short-term conversation memory (last exchanges only)
async def update_conversation_memory(conversation_id, user_msg, bot_msg):
    doc = await conversation_memory.find_one({"_id": conversation_id})

    # New interaction entries (user + assistant)
    entries = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": bot_msg}
    ]

    if doc:
        # Keep only the last 6 messages to limit context size
        msgs = (doc["messages"] + entries)[-6:]
        await conversation_memory.update_one(
            {"_id": conversation_id},
            {"$set": {"messages": msgs, "updatedAt": datetime.utcnow()}}
        )
    else:
        # Create memory document if it does not exist
        await conversation_memory.insert_one({
            "_id": conversation_id,
            "messages": entries,
            "updatedAt": datetime.utcnow()
        })


# Build a textual memory context injected into the LLM prompt
async def get_memory_context(conversation_id):
    doc = await conversation_memory.find_one({"_id": conversation_id})
    if not doc:
        return ""
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in doc["messages"]
    )
