// ================== IMPORTS ==================
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import multer from "multer";

// ================== CONFIG ==================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const uploadDir = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage: multer.memoryStorage() });

dotenv.config({ path: path.resolve(__dirname, "../.env") });

const app = express();
const PORT = process.env.APP_PORTT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000";
const CORS_ORIGINS = process.env.CORS_ORIGINS?.split(",") || ["http://localhost:4200"];

app.use(express.json());
app.use(
  cors({
    origin: CORS_ORIGINS,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
  })
);

// --- Logger global ---
app.use((req, res, next) => {
  console.log(`📥 ${req.method} ${req.url}`);
  next();
});

// ================== ROUTES ==================

// 🔹 1. Récupérer toutes les conversations (relay vers FastAPI)
app.get("/conversations", async (req, res) => {
  console.log("📥 [Express] GET /conversations reçu");
  try {
    const fastApiUrl = "http://127.0.0.1:8000/conversations";
    const response = await axios.get(fastApiUrl);
    console.log("✅ [Express] Réponse reçue de FastAPI :", response.data.length, "conversations");
    res.json(response.data);
  } catch (err) {
    console.error("❌ [Express] Erreur lors du proxy vers FastAPI :", err.message);
    if (err.response) console.error("🔻 FastAPI a répondu :", err.response.data);
    res.status(500).json({ error: "Erreur Express → FastAPI" });
  }
});


// 🔹 2. Créer une nouvelle conversation
app.post("/conversations", async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/conversations`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur création conversation:", err.message);
    res.status(500).json({ error: "Erreur création conversation." });
  }
});

// 🔹 0. Récupérer les messages d’une conversation (proxy vers FastAPI)
app.get("/api/chat/messages/:id", async (req, res) => {
  const { id } = req.params;
  const fastApiUrl = `${FASTAPI_URL}/conversations/${id}/messages`;

  console.log(`📥 [Express] Proxy GET /api/chat/messages/${id}`);
  console.log(`➡️ [Express] Forward vers ${fastApiUrl}`);

  try {
    const response = await axios.get(fastApiUrl);
    console.log(`✅ [Express] ${response.data.length} messages reçus de FastAPI`);
    res.json(response.data);
  } catch (err) {
    console.error(`❌ [Express] Erreur proxy messages: ${err.message}`);
    if (err.response)
      console.error(`🔻 Détail FastAPI: ${err.response.status} ${err.response.statusText}`);
    res.status(500).json({ error: "Erreur proxy messages" });
  }
});


// 🔹 3. Envoyer un message (texte + fichiers)
app.post("/api/chat/message/:id", upload.array("files"), async (req, res) => {
  const { id } = req.params;
  const { text } = req.body;

  console.log(`📥 [Express] Nouveau message pour conversation ${id}`);
  console.log(`📦 [Express] Fichiers reçus : ${req.files?.length || 0}`);
  console.log(`📝 [Express] Texte reçu : "${text || '(vide)'}"`);

  try {
    if (!id) return res.status(400).json({ error: "❌ conv_id manquant" });

    // Créer le FormData pour FastAPI
    const formData = new FormData();
    formData.append("text", text || "");

    // Ajouter les fichiers reçus dans le form-data
    if (Array.isArray(req.files)) {
      for (const f of req.files) {
        formData.append("files", f.buffer, {
          filename: f.originalname,
          contentType: f.mimetype,
          knownLength: f.size,
        });
        console.log(`📎 [Express] Ajout buffer → ${f.originalname} (${f.mimetype}, ${f.size}o)`);
      }
    }
    

    // Proxy vers FastAPI
    const fastApiUrl = `${FASTAPI_URL}/message/${id}`;
    console.log(`➡️ [Express] Envoi vers FastAPI → ${fastApiUrl}`);

    const response = await axios.post(fastApiUrl, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    console.log(`✅ [Express] Message transmis à FastAPI (conversation ${id})`);
    console.log(
      `📄 [Express] Fichiers sauvegardés : ${response.data.files?.length || 0}`
    );

    res.json(response.data);
  } catch (err) {
    console.error("❌ [Express] Erreur /api/chat/message:", err.message);

    if (err.response) {
      console.error(
        `🔻 [FastAPI] ${err.response.status} ${err.response.statusText}`
      );
      console.error("📬 Réponse FastAPI:", err.response.data);
    }

    res.status(500).json({
      error: "Erreur durant l'envoi du message.",
      detail: err.message,
    });
  }
  

  

});


app.put("/api/chat/messages/:id", async (req, res) => {
  const { id } = req.params;
  const { title } = req.body;
  console.log(`✏️ [Express] Requête renommage pour conversation ${id} → ${title}`);

  try {
    const response = await axios.put(`${FASTAPI_URL}/conversations/${id}`, { title });
    res.json(response.data);
  } catch (err) {
    console.error("❌ [Express] Erreur renommage conversation:", err.message);
    res.status(500).json({ error: "Erreur renommage conversation FastAPI." });
  }
});

app.delete("/api/chat/messages/:id", async (req, res) => {
  const { id } = req.params;
  console.log(`🗑️ [Express] Suppression conversation ${id}`);

  try {
    await axios.delete(`${FASTAPI_URL}/conversations/${id}`);
    res.json({ success: true });
  } catch (err) {
    console.error("❌ [Express] Erreur suppression conversation:", err.message);
    res.status(500).json({ error: "Erreur suppression conversation FastAPI." });
  }
});


// 🔹 4. Chat RAG (question / réponse)
app.post("/api/chat", async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/chat`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur /api/chat:", err.message);
    res.status(500).json({ error: "Erreur RAG FastAPI." });
  }
});

// 🔹 Proxy pour télécharger un fichier depuis FastAPI
app.get("/api/chat/file/:id", async (req, res) => {
  const fileId = req.params.id;
  const fastApiUrl = `http://127.0.0.1:8000/file/${fileId}`;

  console.log(`📥 [Express] Requête Angular → /api/chat/file/${fileId}`);
  console.log(`➡️ [Express] Appel FastAPI → ${fastApiUrl}`);

  try {
    const response = await axios.get(fastApiUrl, { responseType: "stream" });

    console.log(`✅ [Express] Fichier trouvé : ${response.headers["content-disposition"] || "(pas de nom)"}`);
    console.log(`📦 [Express] Type MIME : ${response.headers["content-type"]}`);

    res.setHeader("Content-Type", response.headers["content-type"] || "application/octet-stream");
    if (response.headers["content-disposition"])
      res.setHeader("Content-Disposition", response.headers["content-disposition"]);

    response.data.pipe(res);
  } catch (err) {
    console.error(`❌ [Express] Erreur proxy téléchargement : ${err.message}`);
    if (err.response) {
      console.error(`🔻 [Express] Réponse FastAPI : ${err.response.status} ${err.response.statusText}`);
    }
    res.status(502).json({ error: "Erreur proxy téléchargement", detail: err.message });
  }
});



// 🔹 5. Télécharger un fichier (relay direct vers FastAPI)
app.get("/file/:id", async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/file/${req.params.id}`, {
      responseType: "stream",
    });
    res.setHeader("Content-Disposition", response.headers["content-disposition"]);
    res.setHeader("Content-Type", response.headers["content-type"]);
    response.data.pipe(res);
  } catch (err) {
    console.error("❌ Erreur téléchargement fichier:", err.message);
    res.status(500).json({ error: "Erreur récupération fichier FastAPI." });
  }
});

// 🔹 6. Recherche web (relay vers FastAPI)
app.post("/api/websearch", async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/websearch`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error("❌ Erreur /api/websearch:", err.message);
    res.status(500).json({ error: "Erreur recherche web FastAPI." });
  }
});

// ================== START SERVER ==================
app.listen(PORT, () => {
  console.log(`🚀 Express proxy running at http://127.0.0.1:${PORT}`);
  console.log(`🔗 Connected to FastAPI at ${FASTAPI_URL}`);
});
