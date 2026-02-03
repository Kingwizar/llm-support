// ======================================================
// backend/server.js (Express Gateway)
// Step 1: Imports + environment configuration
// ======================================================

// ================== IMPORTS (CommonJS) ==================
// express: HTTP server + routing layer (gateway between clients and FastAPI)
const express = require("express");

// cors: sets CORS headers to allow browser clients (Angular) to call this gateway
const cors = require("cors");

// dotenv: loads environment variables from a .env file
const dotenv = require("dotenv");

// path: filesystem path utilities (safe cross-platform paths)
const path = require("path");

// axios: HTTP client used to forward requests from Express to FastAPI
const axios = require("axios");

// FormData: builds multipart/form-data requests to forward files to FastAPI
const FormData = require("form-data");

// fs: filesystem access (temporary upload storage, cleanup, static hosting)
const fs = require("fs");

// multer: parses multipart/form-data uploads (files coming from Angular/Android)
const multer = require("multer");

// helmet: security headers (CSP, XSS protections, etc.)
const helmet = require("helmet");

// express-rate-limit: basic rate limiting to mitigate abuse / brute force
const rateLimit = require("express-rate-limit");

// csurf: CSRF protection middleware (cookie + header token strategy)
const csrf = require("csurf");

// cookie-parser: reads cookies (session token, csrf cookie)
const cookieParser = require("cookie-parser");

// prom-client: Prometheus metrics client (custom metrics + /metrics endpoint)
const client = require("prom-client");


// ================== PROMETHEUS REGISTRY ==================
// Registry used to store metrics. You expose it later at GET /metrics
const register = new client.Registry();

// Collect default Node.js process metrics (CPU, memory, event loop, etc.)
client.collectDefaultMetrics({ register });


// ================== CONFIG (.env) ==================
// Loads environment variables from ../.env relative to this file
// This is usually used in Docker/Swarm with a bind-mounted env file or secret
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// PORT: Express external port (gateway). Defaults to 3000 if not defined
const PORT = process.env.APP_PORT || 3000;

// FASTAPI_URL: internal URL to the FastAPI service (e.g., http://npone_backend:8000)
const FASTAPI_URL = process.env.FASTAPI_URL;

// AUTH0_DOMAIN: Auth0 tenant domain used in CSP + validation bridge
const AUTH0_DOMAIN =
  process.env.AUTH0_DOMAIN || "dev-5xqrzsdislhri5jj.us.auth0.com";

// APP_ENV: controls security toggles (CSRF cookie secure flag, etc.)
const APP_ENV = process.env.APP_ENV || "dev";

// IS_DEV: internal boolean used to relax some security flags in dev
// In your code: prod is when APP_ENV === "prod"
const IS_DEV = APP_ENV !== "prod";

// Hard fail if the gateway does not know where FastAPI is
if (!FASTAPI_URL) {
  console.error("FASTAPI_URL is missing in .env");
  process.exit(1);
}


// ======================================================
// UPLOAD (multer)
// ======================================================

// Directory used to temporarily store uploaded files
// Files are later forwarded to FastAPI and then deleted
const uploadDir = path.join(__dirname, "uploads");

// Create upload directory if it does not exist
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// Multer storage configuration
// - destination: local temp folder
// - filename: timestamp-based to avoid collisions
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) =>
    cb(null, Date.now() + "-" + file.originalname),
});

// Multer middleware instance
const upload = multer({ storage });


// ======================================================
// EXPRESS INITIALIZATION
// ======================================================

// Create Express application instance
const app = express();

// Trust reverse proxy headers (ngrok, Traefik, Nginx, etc.)
// Required for secure cookies and HTTPS detection
app.set("trust proxy", 1);


// ======================================================
// BODY PARSING + COOKIES
// ======================================================

// Parse JSON bodies (API requests)
app.use(express.json({ limit: "10mb" }));

// Parse URL-encoded form bodies
app.use(express.urlencoded({ extended: true }));

// Enable cookie parsing (session, CSRF, etc.)
app.use(cookieParser());


// ======================================================
// CORS CONFIGURATION
// ======================================================

// Allow cross-origin requests from browser clients
// Required for Angular frontend and credentialed requests
app.use(
  cors({
    origin: true,
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: [
      "Content-Type",
      "Authorization",
      "X-CSRF-Token",
    ],
  })
);


// ======================================================
// HELMET / CONTENT SECURITY POLICY
// ======================================================

// Security headers configuration
// CSP is tuned for Angular + Auth0 + API calls
app.use(
  helmet({
    contentSecurityPolicy: {
      useDefaults: true,
      directives: {
        defaultSrc: ["'self'"],
        baseUri: ["'self'"],
        objectSrc: ["'none'"],

        // Angular runtime scripts (inline allowed)
        scriptSrc: ["'self'", "'unsafe-inline'"],
        scriptSrcAttr: ["'self'", "'unsafe-inline'"],

        // Angular inline styles
        styleSrc: ["'self'", "'unsafe-inline'"],

        // Images and fonts (base64 + blobs)
        imgSrc: ["'self'", "data:", "blob:"],
        fontSrc: ["'self'", "data:"],

        // API calls (FastAPI, Auth0, ngrok)
        connectSrc: [
          "'self'",
          `https://${AUTH0_DOMAIN}`,
          process.env.PUBLIC_BASE_URL || "'self'",
        ],

        // Auth0 iframe and web_message
        frameSrc: ["'self'", `https://${AUTH0_DOMAIN}`],

        formAction: ["'self'"],
        mediaSrc: ["'self'", "blob:"],
      },
    },

    // Required to allow some third-party resources
    crossOriginEmbedderPolicy: false,
  })
);


// ======================================================
// RATE LIMITING
// ======================================================

// Basic protection against abusive clients
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 300,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => req.ip,
  })
);


// ======================================================
// REQUEST LOGGER (BASIC)
// ======================================================

// Simple request logger for debugging
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});


// ======================================================
// AUTH HEADER FORWARDING
// ======================================================

// Store Authorization header for later forwarding to FastAPI
app.use((req, res, next) => {
  req.authHeader = req.headers.authorization || "";
  next();
});


// ======================================================
// HELPER FUNCTIONS
// ======================================================

// Detect HTTPS behind reverse proxy
function isHttps(req) {
  return (
    req.secure ||
    req.headers["x-forwarded-proto"] === "https"
  );
}

// Build Authorization headers for proxy requests
// Priority:
// 1. HttpOnly session cookie (web)
// 2. Authorization header (mobile / fallback)
function authHeaders(req, extra = {}) {
  const h = { ...extra };

  if (req.cookies?.session) {
    h.Authorization = `Bearer ${req.cookies.session}`;
  } else if (req.authHeader) {
    h.Authorization = req.authHeader;
  }

  return h;
}

// Standardized Axios error forwarding
function sendAxiosError(
  res,
  err,
  fallbackStatus = 500,
  fallbackMsg = "Proxy error"
) {
  const status =
    err?.response?.status || fallbackStatus;
  const data = err?.response?.data;

  if (data) return res.status(status).json(data);
  return res
    .status(status)
    .json({ error: fallbackMsg });
}

// Set HttpOnly session cookie after authentication
function setSessionCookie(req, res, token) {
  const secureCookie = isHttps(req);

  res.cookie("session", token, {
    httpOnly: true,
    secure: secureCookie,
    sameSite: "Lax",
    maxAge: 24 * 60 * 60 * 1000,
  });
}


// ======================================================
// DEBUG ENDPOINT
// ======================================================

// Inspect headers, cookies and protocol (useful for auth/debug)
app.get("/debug/headers", (req, res) => {
  res.json({
    https: isHttps(req),
    ip: req.ip,
    cookies: req.cookies || {},
    authHeader: req.headers.authorization || null,
  });
});


// ======================================================
// PROMETHEUS METRICS (IN-FLIGHT REQUESTS)
// ======================================================

// Gauge tracking active HTTP requests
const httpInFlight = new client.Gauge({
  name: "http_requests_in_flight",
  help: "Number of in-flight HTTP requests",
  labelNames: ["method", "route"],
});

// Register metric
register.registerMetric(httpInFlight);

// Structured request logging with timing
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



// ======================================================
// AUTH ROUTES (NO CSRF HERE)
// These routes are intentionally excluded from CSRF
// because they are entry points for authentication
// ======================================================

// ------------------------------------------------------
// POST /auth/login
// - Forwards credentials to FastAPI
// - Creates a session cookie for WEB clients
// - Returns a token directly for MOBILE clients
// ------------------------------------------------------
app.post("/auth/login", async (req, res) => {
  try {
    // Forward login credentials to FastAPI
    const r = await axios.post(
      `${FASTAPI_URL}/auth/login`,
      req.body
    );

    // Always create a session cookie (used by web clients)
    setSessionCookie(req, res, r.data.access_token);

    // Detect mobile clients (Android)
    const isMobile =
      req.headers["x-client-type"] === "android" ||
      req.headers["user-agent"]?.toLowerCase().includes("okhttp");

    if (isMobile) {
      // Mobile clients handle the token themselves
      res.json({
        access_token: r.data.access_token,
        token_type: "bearer",
      });
    } else {
      // Web clients rely on HttpOnly cookie
      res.json({ success: true });
    }
  } catch (err) {
    sendAxiosError(res, err, 401, "Login failed");
  }
});


// ------------------------------------------------------
// POST /auth/register
// - Forwards user registration to FastAPI
// ------------------------------------------------------
app.post("/auth/register", async (req, res) => {
  try {
    const r = await axios.post(
      `${FASTAPI_URL}/auth/register`,
      req.body
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 400, "Register failed");
  }
});


// ------------------------------------------------------
// POST /auth/auth0
// - Bridge between Auth0 access token and backend session
// - Validates token via FastAPI
// - Converts it into an HttpOnly cookie
// ------------------------------------------------------
app.post("/auth/auth0", async (req, res) => {
  const authHeader = req.headers.authorization;

  // Require Bearer token
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Missing Authorization" });
  }

  try {
    // Validate Auth0 token through FastAPI
    await axios.get(`${FASTAPI_URL}/auth/me`, {
      headers: { Authorization: authHeader },
    });

    // Convert Auth0 token into backend session cookie
    const token = authHeader.replace("Bearer ", "").trim();
    setSessionCookie(req, res, token);

    res.json({ success: true });
  } catch (e) {
    return res.status(401).json({ error: "Invalid Auth0 token" });
  }
});


// ======================================================
// CSRF PROTECTION
// - Applied only to WEB clients
// - Excluded for auth routes and Android clients
// ======================================================

// CSRF middleware configuration
const csrfProtection = csrf({
  cookie: {
    key: "_csrf",
    httpOnly: false, // Angular may need to read it
    secure: !IS_DEV, // secure cookies in production
    sameSite: "Lax",
  },
});

// Conditional CSRF application
app.use((req, res, next) => {
  // Skip CSRF for authentication routes
  if (req.path.startsWith("/auth/")) return next();

  // Skip CSRF for Android clients
  const isAndroid =
    req.headers["x-client-type"] === "android" ||
    req.headers["user-agent"]?.toLowerCase().includes("okhttp");

  if (isAndroid) return next();

  // Apply CSRF protection for web clients
  return csrfProtection(req, res, next);
});


// ------------------------------------------------------
// GET /csrf-token
// - Provides a CSRF token for Angular
// ------------------------------------------------------
app.get("/csrf-token", csrfProtection, (req, res) => {
  res.json({ csrfToken: req.csrfToken() });
});


// ======================================================
// PROTECTED ROUTES (AUTH REQUIRED)
// ======================================================

// ------------------------------------------------------
// GET /auth/me
// - Returns authenticated user info
// ------------------------------------------------------
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


// ======================================================
// CONVERSATIONS (PROXY TO FASTAPI)
// ======================================================

// List user conversations
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

// Create a new conversation
app.post("/conversations", async (req, res) => {
  try {
    const r = await axios.post(
      `${FASTAPI_URL}/conversations`,
      req.body,
      { headers: authHeaders(req) }
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// Rename a conversation
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

// Delete a conversation
app.delete("/conversations/:id", async (req, res) => {
  try {
    const r = await axios.delete(
      `${FASTAPI_URL}/conversations/${req.params.id}`,
      { headers: authHeaders(req) }
    );
    res.json(r.data || { success: true });
  } catch (err) {
    sendAxiosError(res, err, 400, "Delete failed");
  }
});


// ======================================================
// LEGACY /api ROUTES (BACKWARD COMPATIBILITY)
// ======================================================

// Delete conversation (legacy)
app.delete("/api/chat/messages/:id", async (req, res) => {
  try {
    const r = await axios.delete(
      `${FASTAPI_URL}/conversations/${req.params.id}`,
      { headers: authHeaders(req) }
    );
    res.json(r.data || { success: true });
  } catch (err) {
    sendAxiosError(res, err, 400, "Delete failed");
  }
});

// Rename conversation (legacy)
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

// Get messages (legacy)
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


// ======================================================
// MESSAGE + FILE UPLOAD (PROXY)
// ======================================================

// Forward message + uploaded files to FastAPI
async function forwardMessageToFastAPI(req, res) {
  const uploaded = Array.isArray(req.files) ? req.files : [];

  try {
    // Build multipart request
    const formData = new FormData();
    formData.append("text", req.body.text || "");

    for (const f of uploaded) {
      formData.append(
        "files",
        fs.readFileSync(f.path),
        {
          filename: f.originalname,
          contentType: f.mimetype,
        }
      );
    }

    // Forward request to FastAPI
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
    // Cleanup temporary uploaded files
    for (const f of uploaded) {
      try {
        fs.unlinkSync(f.path);
      } catch (_) {}
    }
  }
}

// Web legacy endpoint
app.post(
  "/api/chat/message/:id",
  upload.array("files"),
  forwardMessageToFastAPI
);

// Android alias
app.post(
  "/message/:id",
  upload.array("files"),
  forwardMessageToFastAPI
);

// ======================================================
// CHAT (RAG) PROXY
// ======================================================

// ------------------------------------------------------
// POST /api/chat
// - Main RAG chat endpoint for WEB clients
// - Forwards the request body to FastAPI /chat
// ------------------------------------------------------
app.post("/api/chat", async (req, res) => {
  try {
    const r = await axios.post(
      `${FASTAPI_URL}/chat`,
      req.body,
      { headers: authHeaders(req) }
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});

// ------------------------------------------------------
// POST /chat
// - Alias endpoint for ANDROID clients
// ------------------------------------------------------
app.post("/chat", async (req, res) => {
  try {
    const r = await axios.post(
      `${FASTAPI_URL}/chat`,
      req.body,
      { headers: authHeaders(req) }
    );
    res.json(r.data);
  } catch (err) {
    sendAxiosError(res, err, 401, "Unauthorized");
  }
});


// ======================================================
// FILE DOWNLOAD PROXY
// ======================================================

// ------------------------------------------------------
// GET /api/chat/file/:id
// - Streams a file from FastAPI to the client
// - Preserves Content-Type and Content-Disposition
// ------------------------------------------------------
app.get("/api/chat/file/:id", async (req, res) => {
  try {
    const r = await axios({
      url: `${FASTAPI_URL}/file/${req.params.id}`,
      method: "GET",
      responseType: "stream",
      headers: authHeaders(req),
    });

    // Forward file headers
    res.setHeader(
      "Content-Type",
      r.headers["content-type"] || "application/octet-stream"
    );
    res.setHeader(
      "Content-Disposition",
      r.headers["content-disposition"] || "attachment"
    );

    // Pipe FastAPI stream directly to client
    r.data.pipe(res);
  } catch (err) {
    sendAxiosError(res, err, 500, "File download failed");
  }
});


// ======================================================
// PROMETHEUS METRICS
// ======================================================

// ------------------------------------------------------
// GET /metrics
// - Exposes Node.js + custom metrics for Prometheus
// ------------------------------------------------------
app.get("/metrics", async (req, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});


// ======================================================
// STATIC ANGULAR FRONTEND
// ======================================================

// Path to Angular production build
const angularDist = path.join(
  __dirname,
  "dist/frontend-angular/browser"
);

// Serve Angular static assets
app.use(express.static(angularDist));
app.use(express.static(angularDist));


// ------------------------------------------------------
// API ROUTE SAFETY NET
// - Prevents Angular routing from hijacking API paths
// ------------------------------------------------------
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
    return res
      .status(404)
      .json({ error: "API route not found" });
  }
  next();
});


// ------------------------------------------------------
// Angular SPA fallback
// - All non-API routes return index.html
// ------------------------------------------------------
app.use((req, res) => {
  res.sendFile(path.join(angularDist, "index.html"));
});


// ======================================================
// GLOBAL ERROR HANDLER
// ======================================================

app.use((err, req, res, next) => {
  console.error("Express error:", err);
  res.status(500).json({
    error: "Internal server error",
    detail: String(err?.message || err),
  });
});


// ======================================================
// START SERVER
// ======================================================

// Start Express gateway
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Express running → http://0.0.0.0:${PORT}`);
  console.log(`Connected FastAPI → ${FASTAPI_URL}`);
  console.log(`APP_ENV=${APP_ENV} (IS_DEV=${IS_DEV})`);
});
