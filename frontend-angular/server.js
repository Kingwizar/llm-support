// ================== IMPORTS (CommonJS) ==================
const express = require('express');
const cors = require('cors');
const dotenv = require("dotenv");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const multer = require("multer");

// ================== CONFIG ==================
dotenv.config({ path: path.resolve(__dirname, "../.env") });

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage });

// Express init
const app = express();
const PORT = process.env.APP_PORT;
const FASTAPI_URL = process.env.FASTAPI_URL;
const CORS_ORIGINS = process.env.CORS_ORIGINS?.split(",") || ["http://localhost:4200"];

app.use(express.json());
app.use(
  cors({
    origin: CORS_ORIGINS,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
  })
);

// Logger
app.use((req, res, next) => {
  console.log(`📥 ${req.method} ${req.url}`);
  next();
});

// ================== ROUTES ==================

// 1. Récupérer toutes les conversations
app.get("/conversations", async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/conversations`);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur conversations:", err.message);
    res.status(500).json({ error: "Erreur Express → FastAPI" });
  }
});

// 2. Nouvelle conversation
app.post("/conversations", async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/conversations`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur création:", err.message);
    res.status(500).json({ error: "Erreur création conversation." });
  }
});

// Messages
app.get("/api/chat/messages/:id", async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/conversations/${req.params.id}/messages`);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur messages:", err.message);
    res.status(500).json({ error: "Erreur proxy messages" });
  }
});
// 📁 Télécharger un fichier depuis FastAPI via proxy Express
app.get("/api/chat/file/:id", async (req, res) => {
  const fileId = req.params.id;

  try {
    const url = `${FASTAPI_URL}/file/${fileId}`;
    console.log("🔗 Proxy téléchargement →", url);

    const response = await axios({
      url,
      method: "GET",
      responseType: "stream"
    });

    res.setHeader("Content-Type", response.headers["content-type"]);
    res.setHeader("Content-Disposition", response.headers["content-disposition"] || "attachment");

    response.data.pipe(res);

  } catch (err) {
    console.error("❌ Erreur proxy fichier:", err.message);
    res.status(500).json({ error: "Impossible de récupérer le fichier." });
  }
});


// Message + fichiers
app.post("/api/chat/message/:id", upload.array("files"), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append("text", req.body.text || "");

    if (Array.isArray(req.files)) {
      for (const f of req.files) {
        formData.append("files", fs.readFileSync(f.path), {
          filename: f.originalname,
          contentType: f.mimetype,
        });
      }
    }

    const response = await axios.post(
      `${FASTAPI_URL}/message/${req.params.id}`,
      formData,
      { headers: formData.getHeaders() }
    );

    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur message:", err.message);
    res.status(500).json({ error: "Erreur envoi message." });
  }
});

// Renommage
app.put("/api/chat/messages/:id", async (req, res) => {
  try {
    const response = await axios.put(`${FASTAPI_URL}/conversations/${req.params.id}`, {
      title: req.body.title,
    });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: "Erreur renommage conversation" });
  }
});

// Suppression conversation
app.delete("/api/chat/messages/:id", async (req, res) => {
  try {
    await axios.delete(`${FASTAPI_URL}/conversations/${req.params.id}`);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: "Erreur suppression conversation" });
  }
});

// Chat RAG
app.post("/api/chat", async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/chat`, req.body);
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: "Erreur RAG FastAPI." });
  }
});

// ================== START SERVER ==================
app.listen(PORT, () => {
  console.log(`🚀 Express running → http://127.0.0.1:${PORT}`);
  console.log(`🔗 Connected FastAPI → ${FASTAPI_URL}`);
});