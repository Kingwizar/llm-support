// ================== IMPORTS (CommonJS) ==================
const express = require("express");
const cors = require("cors");
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

// ================== EXPRESS INIT ==================
const app = express();
const PORT = process.env.APP_PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL;

if (!FASTAPI_URL) {
  console.error("❌ FASTAPI_URL is missing in .env");
  process.exit(1);
}

const CORS_ORIGINS = process.env.CORS_ORIGINS?.split(",") || [
  "http://localhost:4200",
];

app.use(express.json());
app.use(
  cors({
    origin: CORS_ORIGINS,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
  })
);

// ================== LOGGER ==================
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// ================== AUTH HEADER FORWARD ==================
app.use((req, res, next) => {
  req.authHeader = req.headers.authorization || "";
  next();
});

// ================== HELPERS ==================
function authHeaders(req, extra = {}) {
  // Ne pas envoyer Authorization vide (certains serveurs n'aiment pas)
  const h = { ...extra };
  if (req.authHeader) h.Authorization = req.authHeader;
  return h;
}

function sendAxiosError(res, err, fallbackStatus = 500, fallbackMsg = "Proxy error") {
  const status = err?.response?.status || fallbackStatus;
  const data = err?.response?.data;

  // renvoyer l'erreur FastAPI si possible
  if (data) return res.status(status).json(data);

  return res.status(status).json({ error: fallbackMsg });
}

// ================== AUTH ROUTES ==================
app.post("/auth/login", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/auth/login`, req.body);
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Login failed");
  }
});

app.post("/auth/register", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/auth/register`, req.body);
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 400, "Register failed");
  }
});

app.get("/auth/me", async (req, res) => {
  try {
    const r = await axios.get(`${FASTAPI_URL}/auth/me`, {
      headers: authHeaders(req),
    });
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// ================== CONVERSATIONS ==================
app.get("/conversations", async (req, res) => {
  try {
    const r = await axios.get(`${FASTAPI_URL}/conversations`, {
      headers: authHeaders(req),
    });
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

app.post("/conversations", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/conversations`, req.body, {
      headers: authHeaders(req),
    });
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// Renommage (ton Angular appelle /api/chat/messages/:id)
app.put("/api/chat/messages/:id", async (req, res) => {
  try {
    const r = await axios.put(
      `${FASTAPI_URL}/conversations/${req.params.id}`,
      { title: req.body.title },
      { headers: authHeaders(req) }
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 400, "Rename failed");
  }
});

// Suppression (ton Angular appelle /api/chat/messages/:id)
app.delete("/api/chat/messages/:id", async (req, res) => {
  try {
    const r = await axios.delete(`${FASTAPI_URL}/conversations/${req.params.id}`, {
      headers: authHeaders(req),
    });
    res.json(r.data || { success: true });
  } catch (err) {
    sendAxiosError(res, err, 400, "Delete failed");
  }
});

app.delete("/conversations/:id", async (req, res) => {
  try {
    const r = await axios.delete(
      `${FASTAPI_URL}/conversations/${req.params.id}`,
      {
        headers: {
          Authorization: req.authHeader
        }
      }
    );

    res.json(r.data);
  } catch (e) {
    res
      .status(e.response?.status || 500)
      .json(e.response?.data || { error: "Delete failed" });
  }
});


// ✅ IMPORTANT: on SUPPRIME la route doublon suivante:
// app.delete("/conversations/:id", ...)  <-- elle faisait doublon et pouvait créer des comportements bizarres

// ================== MESSAGES ==================
app.get("/api/chat/messages/:id", async (req, res) => {
  try {
    const r = await axios.get(
      `${FASTAPI_URL}/conversations/${req.params.id}/messages`,
      { headers: authHeaders(req) }
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

app.put("/conversations/:id", async (req, res) => {
  try {
    const r = await axios.put(
      `${FASTAPI_URL}/conversations/${req.params.id}`,
      { title: req.body.title },
      {
        headers: {
          Authorization: req.authHeader
        }
      }
    );

    res.json(r.data);
  } catch (e) {
    res
      .status(e.response?.status || 500)
      .json(e.response?.data || { error: "Rename failed" });
  }
});


app.post("/api/chat/message/:id", upload.array("files"), async (req, res) => {
  const uploaded = Array.isArray(req.files) ? req.files : [];

  try {
    const formData = new FormData();
    formData.append("text", req.body.text || "");

    for (const f of uploaded) {
      formData.append("files", fs.readFileSync(f.path), {
        filename: f.originalname,
        contentType: f.mimetype,
      });
    }

    const r = await axios.post(`${FASTAPI_URL}/message/${req.params.id}`, formData, {
      headers: authHeaders(req, formData.getHeaders()),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });

    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 500, "Message send failed");
  } finally {
    // ✅ Nettoyage des fichiers temporaires
    for (const f of uploaded) {
      try { fs.unlinkSync(f.path); } catch (_) {}
    }
  }
});

// ================== CHAT RAG ==================
app.post("/api/chat", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/chat`, req.body, {
      headers: authHeaders(req),
    });
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// ================== FILE DOWNLOAD ==================
app.get("/api/chat/file/:id", async (req, res) => {
  try {
    const r = await axios({
      url: `${FASTAPI_URL}/file/${req.params.id}`,
      method: "GET",
      responseType: "stream",
      headers: authHeaders(req),
    });

    res.setHeader("Content-Type", r.headers["content-type"] || "application/octet-stream");
    res.setHeader(
      "Content-Disposition",
      r.headers["content-disposition"] || "attachment"
    );

    r.data.pipe(res);
  } catch (err) {
    sendAxiosError(res, err, 500, "File download failed");
  }
});

// ================== START SERVER ==================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Express running → http://0.0.0.0:${PORT}`);
  console.log(`Connected FastAPI → ${FASTAPI_URL}`);
});
