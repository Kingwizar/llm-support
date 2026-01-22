// backend/server.js (Express Gateway) — CORRIGÉ COMPLET


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
const client = require("prom-client");

const register = new client.Registry();
client.collectDefaultMetrics({ register });
// ================== CONFIG ==================
dotenv.config({ path: path.resolve(__dirname, "../.env") });

const PORT = process.env.APP_PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL;
const AUTH0_DOMAIN =
  process.env.AUTH0_DOMAIN || "dev-5xqrzsdislhri5jj.us.auth0.com";

const APP_ENV = process.env.APP_ENV || "dev";
const IS_DEV = APP_ENV !== "prod";

if (!FASTAPI_URL) {
  console.error("❌ FASTAPI_URL is missing in .env");
  process.exit(1);
}

// ================== UPLOAD ==================
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage });

// ================== EXPRESS INIT ==================
const app = express();

app.set("trust proxy", 1);

// ================== BODY + COOKIES ==================
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// ================== CORS ==================
app.use(
  cors({
    origin: true,
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-CSRF-Token"],
  })
);

// ================== HELMET / CSP ==================
app.use(
  helmet({
    contentSecurityPolicy: {
      useDefaults: true,
      directives: {
        defaultSrc: ["'self'"],
        baseUri: ["'self'"],
        objectSrc: ["'none'"],

        // ✅ Angular + certains attributs runtime
        scriptSrc: ["'self'", "'unsafe-inline'"],
        scriptSrcAttr: ["'self'", "'unsafe-inline'"],

        styleSrc: ["'self'", "'unsafe-inline'"],

        imgSrc: ["'self'", "data:", "blob:"],
        fontSrc: ["'self'", "data:"],

        // ✅ Auth0 XHR/fetch + backend
        connectSrc: [
          "'self'",
          `https://${AUTH0_DOMAIN}`,
          process.env.PUBLIC_BASE_URL || "'self'"
        ],

        // ✅ Auth0 iframe/web_message
        frameSrc: ["'self'", `https://${AUTH0_DOMAIN}`],

        formAction: ["'self'"],
        mediaSrc: ["'self'", "blob:"],
      },
    },
    crossOriginEmbedderPolicy: false,
  })
);

// ================== RATE LIMIT ==================
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 300,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => req.ip,
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
function isHttps(req) {
  return req.secure || req.headers["x-forwarded-proto"] === "https";
}

function authHeaders(req, extra = {}) {
  const h = { ...extra };

  // cookie session prioritaire (HttpOnly)
  if (req.cookies?.session) {
    h.Authorization = `Bearer ${req.cookies.session}`;
  } else if (req.authHeader) {
    // fallback (si token envoyé par header)
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

function setSessionCookie(req, res, token) {
  //  sur ngrok (https) => secure MUST be true
  const secureCookie = isHttps(req);

  res.cookie("session", token, {
    httpOnly: true,
    secure: secureCookie,
    sameSite: "Lax",
    maxAge: 24 * 60 * 60 * 1000,
  });
}

// ================== DEBUG (OPTIONNEL MAIS UTILE) ==================
app.get("/debug/headers", (req, res) => {
  res.json({
    https: isHttps(req),
    ip: req.ip,
    cookies: req.cookies || {},
    authHeader: req.headers.authorization || null,
  });
});

const httpInFlight = new client.Gauge({
  name: "http_requests_in_flight",
  help: "Nombre de requêtes HTTP en cours",
  labelNames: ["method", "route"],
});
register.registerMetric(httpInFlight);

app.use((req, res, next) => {
  const start = Date.now();

  res.on("finish", () => {
    const log = {
      ts: new Date().toISOString(),
      method: req.method,
      path: req.originalUrl,
      status: res.statusCode,
      duration_ms: Date.now() - start,
      ip: req.ip,
      user_agent: req.headers["user-agent"],
    };

    console.log(JSON.stringify(log));
  });

  next();
});




// ================== AUTH ROUTES (⚠️ PAS DE CSRF ICI) ==================
app.post("/auth/login", async (req, res) => {
  try {
    const r = await axios.post(`${FASTAPI_URL}/auth/login`, req.body);

    //  Cookie session pour WEB
    setSessionCookie(req, res, r.data.access_token);

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
      // 🌐 WEB : cookie HttpOnly
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

//  BRIDGE AUTH0 → SESSION (⚠️ PAS DE CSRF ICI)
app.post("/auth/auth0", async (req, res) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Missing Authorization" });
  }

  try {
    //  On valide le token Auth0 en appelant FastAPI /auth/me
    await axios.get(`${FASTAPI_URL}/auth/me`, {
      headers: { Authorization: authHeader },
    });

    //  Créer la session backend (cookie)
    const token = authHeader.replace("Bearer ", "").trim();
    setSessionCookie(req, res, token);

    res.json({ success: true });
  } catch (e) {
    return res.status(401).json({ error: "Invalid Auth0 token" });
  }
});

// ================== CSRF (PROTECTED) ==================
//  CSRF appliqué à tout le reste (pas aux routes /auth/*)
const csrfProtection = csrf({
  cookie: {
    key: "_csrf",
    httpOnly: false, //  Angular doit pouvoir le lire si besoin (mais tu l’envoies via endpoint)
    secure: !IS_DEV, // prod true (si tu mets APP_ENV=prod)
    sameSite: "Lax",
  },
});

//  middleware conditionnel : exclure /auth/*
app.use((req, res, next) => {
  // ❌ Pas de CSRF pour /auth/*
  if (req.path.startsWith("/auth/")) return next();

  // ❌ Pas de CSRF pour Android
  const isAndroid =
    req.headers["x-client-type"] === "android" ||
    req.headers["user-agent"]?.toLowerCase().includes("okhttp");

  if (isAndroid) return next();

  // ✅ CSRF uniquement pour Web
  return csrfProtection(req, res, next);
});


// Endpoint pour Angular: récupérer un token CSRF
app.get("/csrf-token", csrfProtection,(req, res) => {
  res.json({ csrfToken: req.csrfToken() });
});

// ================== PROTECTED ROUTES ==================
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

// ================== WEB legacy /api ==================
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

//  Alias Android
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

// ================== MESSAGE + FILES ==================
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

    const r = await axios.post(`${FASTAPI_URL}/message/${req.params.id}`, formData, {
      headers: authHeaders(req, formData.getHeaders()),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });

    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 500, "Message send failed");
  } finally {
    for (const f of uploaded) {
      try {
        fs.unlinkSync(f.path);
      } catch (_) {}
    }
  }
}

// WEB legacy /api
app.post("/api/chat/message/:id", upload.array("files"), forwardMessageToFastAPI);

//  Alias Android
app.post("/message/:id", upload.array("files"), forwardMessageToFastAPI);

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

//  Alias Android
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
    res.setHeader("Content-Disposition", r.headers["content-disposition"] || "attachment");
    r.data.pipe(res);
  } catch (err) {
    sendAxiosError(res, err, 500, "File download failed");
  }
});

app.get("/metrics", async (req, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});

// ================== STATIC ANGULAR ==================
const angularDist = path.join(__dirname, "dist/frontend-angular/browser");app.use(express.static(angularDist));
app.use(express.static(angularDist));

app.use((req, res, next) => {
  if (
    req.path.startsWith("/api") ||
    req.path.startsWith("/auth") ||
    req.path.startsWith("/conversations") ||
    req.path.startsWith("/chat") ||
    req.path.startsWith("/message") ||
    req.path.startsWith("/csrf-token") ||
    req.path.startsWith("/debug")
  ) {
    return res.status(404).json({ error: "API route not found" });
  }
  next();
});

app.use((req, res) => {
  res.sendFile(path.join(angularDist, "index.html"));
});
app.use((err, req, res, next) => {
  console.error("🔥 Express error:", err);
  res.status(500).json({ error: "Internal server error", detail: String(err?.message || err) });
});

// ================== START SERVER ==================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Express running → http://0.0.0.0:${PORT}`);
  console.log(`Connected FastAPI → ${FASTAPI_URL}`);
  console.log(`APP_ENV=${APP_ENV} (IS_DEV=${IS_DEV})`);
});
