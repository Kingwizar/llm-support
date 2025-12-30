// backend/server.js (Express Gateway) — version safe (ne casse rien)
// - Garde toutes tes routes /api existantes
// - AJOUTE des alias sans /api pour Android (/chat, /message/:id, /conversations/:id/messages)
// - Supprime le doublon /conversations/:id (DELETE) pour éviter comportements bizarres

// ================== IMPORTS (CommonJS) ==================
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const multer = require("multer");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const csrf = require("csurf");
const cookieParser = require("cookie-parser");
const IS_DEV = process.env.APP_ENV !== "prod";


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
app.set("trust proxy", true);

const PORT = process.env.APP_PORT;
const FASTAPI_URL = process.env.FASTAPI_URL;

if (!FASTAPI_URL) {
  console.error("❌ FASTAPI_URL is missing in .env");
  process.exit(1);
}



// ================== BODY + COOKIES ==================
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());


app.use(
  cors({
    origin: true, // ✅ accepte dynamiquement l’origine (ngrok)
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-CSRF-Token"],
  })
);


// ================== HELMET / CSP ==================
// Fix Auth0 bloqué: il faut autoriser Auth0 dans frame-src + connect-src
// et autoriser style-src (Angular charge parfois des styles)
// On laisse script-src 'self' (pas de inline scripts)
const AUTH0_DOMAIN = process.env.AUTH0_DOMAIN || "dev-5xqrzsdislhri5jj.us.auth0.com";

app.use(
  helmet({
    contentSecurityPolicy: {
      useDefaults: true,
      directives: {
        defaultSrc: ["'self'"],
        baseUri: ["'self'"],
        objectSrc: ["'none'"],

        // Angular bundles
        scriptSrcAttr: ["'self'", "'unsafe-inline'"],

        styleSrc: ["'self'", "'unsafe-inline'"],

        imgSrc: ["'self'", "data:", "blob:"],
        fontSrc: ["'self'", "data:"],

        // Requêtes XHR/fetch vers ton propre domaine + Auth0
        connectSrc: ["'self'", `https://${AUTH0_DOMAIN}`],

        // Auth0 utilise parfois des iframes / web_message
        frameSrc: ["'self'", `https://${AUTH0_DOMAIN}`],

        // Form action
        formAction: ["'self'"],

        // Media (si audio blob)
        mediaSrc: ["'self'", "blob:"],
      },
    },
    crossOriginEmbedderPolicy: false, // évite certains blocages avec iframes/ressources
  })
);

// ================== RATE LIMIT ==================
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 300,
    standardHeaders: true,
    legacyHeaders: false,
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
  const h = { ...extra };

  // cookie session prioritaire (HttpOnly)
  if (req.cookies?.session) {
    h.Authorization = `Bearer ${req.cookies.session}`;
  } else if (req.authHeader) {
    // fallback legacy
    h.Authorization = req.authHeader;
  }

  return h;
}

function sendAxiosError(res, err, fallbackStatus = 500, fallbackMsg = "Proxy error") {
  const status = err?.response?.status || fallbackStatus;
  const data = err?.response?.data;
  if (data) return res.status(status).json(data);
  return res.status(status).json({ error: fallbackMsg });
}

// ================== AUTH ROUTES ==================
app.post("/auth/login", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/auth/login`, req.body);

    // NOTE: secure:true => nécessite HTTPS (ngrok OK)
    res.cookie("session", r.data.access_token, {
      httpOnly: true,
      secure: true,
      sameSite: "Lax", // plus compatible Auth0/redirects que Strict
      maxAge: 24 * 60 * 60 * 1000,
    });

    const isMobile =
  req.headers["x-client-type"] === "android" ||
  req.headers["user-agent"]?.toLowerCase().includes("okhttp");

if (isMobile) {
  // 📱 ANDROID : retourne le token
  res.json({
    access_token: r.data.access_token,
    token_type: "bearer",
  });
} else {
  // 🌐 WEB : cookie HttpOnly uniquement
  res.json({ success: true });
}
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

// =====================================================
// ================== CSRF (PROTECTED) =================
// =====================================================
// On active CSRF après login/register
const csrfProtection = csrf({
  cookie: {
    key: "_csrf",
    httpOnly: false,           // ✅ OBLIGATOIRE
    secure: !IS_DEV,           // false en dev, true en prod/ngrok
    sameSite: "Lax",
  },
});

app.use(csrfProtection);

// Endpoint pour Angular: récupérer un token CSRF
app.get("/csrf-token", (req, res) => {
  res.json({ csrfToken: req.csrfToken() });
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

app.put("/conversations/:id", async (req, res) => {
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

app.delete("/conversations/:id", async (req, res) => {
  try {
    const r = await axios.delete(`${FASTAPI_URL}/conversations/${req.params.id}`, {
      headers: authHeaders(req),
    });
    res.json(r.data || { success: true });
  } catch (err) {
    sendAxiosError(res, err, 400, "Delete failed");
  }
});

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

// ================== MESSAGES (WEB legacy /api) ==================
// (ton Angular appelle /api/chat/messages/:id)
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

// ✅ Alias Android (ne casse rien)
// Android appelle /conversations/{id}/messages
app.get("/conversations/:id/messages", async (req, res) => {
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

// ================== MESSAGES (WEB legacy /api) ==================
// Angular appelle /api/chat/messages/:id
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


async function forwardMessageToFastAPI(req, res) {
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

    const r = await axios.post(
      `${FASTAPI_URL}/message/${req.params.id}`,
      formData,
      {
        headers: authHeaders(req, formData.getHeaders()),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
      }
    );

    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 500, "Message send failed");
  } finally {
    for (const f of uploaded) {
      try { fs.unlinkSync(f.path); } catch (_) {}
    }
  }
}


// ================== SEND MESSAGE + FILES (WEB legacy /api) ==================
app.post("/api/chat/message/:id", upload.array("files"), forwardMessageToFastAPI);

// ✅ Alias Android (ne casse rien)
// Android appelle /message/{id}
app.post("/message/:id", upload.array("files"), forwardMessageToFastAPI);


// ================== CHAT RAG (WEB legacy /api) ==================
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

// ✅ Alias Android (ne casse rien)
// Android appelle /chat
app.post("/chat", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/chat`, req.body, {
      headers: authHeaders(req),
    });
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// =====================================================
// ================== FILE DOWNLOAD (WEB) ==============
// =====================================================
app.get("/api/chat/file/:id", async (req, res) => {
  try {
    const r = await axios({
      url: `${FASTAPI_URL}/file/${req.params.id}`,
      method: "GET",
      responseType: "stream",
      headers: authHeaders(req),
    });

    res.setHeader("Content-Type", r.headers["content-type"] || "application/octet-stream");
    res.setHeader("Content-Disposition", r.headers["content-disposition"] || "attachment");

    r.data.pipe(res);
  } catch (err) {
    sendAxiosError(res, err, 500, "File download failed");
  }
});

// =====================================================
// ================== STATIC ANGULAR ===================
// =====================================================
const angularDist = path.join(__dirname, "../frontend-angular/dist/frontend-angular/browser");
app.use(express.static(angularDist));

app.use((req, res, next) => {
  if (
    req.path.startsWith("/api") ||
    req.path.startsWith("/auth") ||
    req.path.startsWith("/conversations") ||
    req.path.startsWith("/chat") ||
    req.path.startsWith("/message") ||
    req.path.startsWith("/csrf-token")
  ) {
    return res.status(404).json({ error: "API route not found" });
  }
  next();
});


app.use((req, res) => {
  res.sendFile(path.join(angularDist, "index.html"));
});

// ================== START SERVER ==================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Express running → http://0.0.0.0:${PORT}`);
  console.log(`Connected FastAPI → ${FASTAPI_URL}`);
});
