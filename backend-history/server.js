// ================== IMPORTS ==================
import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";

// ================== CONFIG ==================
dotenv.config(); // ← charge le .env depuis la racine

const app = express();
const PORT = process.env.APP_PORT || 3000;
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/chatdb";

// ================== MIDDLEWARES ==================
app.use(express.json());

// --- CORS sécurisé (Angular) ---
app.use(
  cors({
    origin: ["http://localhost:4200", "http://127.0.0.1:4200"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
  })
);
app.options(/.*/, cors()); // ✅ Express 5 compatible (remplace "*")

// --- Logger global ---
app.use((req, res, next) => {
  console.log(`📥 ${req.method} ${req.url}`);
  next();
});

// ================== MONGODB ==================
mongoose
  .connect(MONGO_URI)
  .then(() => console.log(`✅ Connected to MongoDB at ${MONGO_URI}`))
  .catch((err) => console.error("❌ MongoDB error:", err));

// ================== MODELS ==================
const MessageSchema = new mongoose.Schema({
  role: String, // "user" ou "bot"
  content: String,
});

const ConversationSchema = new mongoose.Schema({
  title: String,
  messages: [MessageSchema],
});

const Conversation = mongoose.model("Conversation", ConversationSchema);

// ================== ROUTES ==================

// 🔹 1. Récupérer toutes les conversations
app.get("/conversations", async (req, res) => {
  try {
    const convos = await Conversation.find();
    console.log(`🧠 Conversations trouvées : ${convos.length}`);
    res.json(convos);
  } catch (err) {
    console.error("❌ Erreur get /conversations :", err);
    res.status(500).json({ error: err.message });
  }
});

// 🔹 2. Créer une nouvelle conversation
app.post("/conversations", async (req, res) => {
  try {
    const convo = new Conversation({ title: req.body.title, messages: [] });
    await convo.save();
    console.log(`✨ Nouvelle conversation : ${convo.title}`);
    res.json(convo);
  } catch (err) {
    console.error("❌ Erreur création conversation :", err);
    res.status(500).json({ error: err.message });
  }
});

// 🔹 3. Renommer une conversation
app.put("/conversations/:id", async (req, res) => {
  try {
    const convo = await Conversation.findByIdAndUpdate(
      req.params.id,
      { title: req.body.title },
      { new: true }
    );
    console.log(`✏️ Conversation renommée : ${convo.title}`);
    res.json(convo);
  } catch (err) {
    console.error("❌ Erreur renommage :", err);
    res.status(500).json({ error: err.message });
  }
});

// 🔹 4. Supprimer une conversation
app.delete("/conversations/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await Conversation.findByIdAndDelete(id);
    console.log(`🗑️ Conversation supprimée : ${id}`);
    res.json({ success: true });
  } catch (err) {
    console.error("❌ Erreur suppression :", err);
    res.status(500).json({ error: err.message });
  }
});

// 🔹 5. Ajouter un message
app.post("/conversations/:id/messages", async (req, res) => {
  try {
    const convo = await Conversation.findById(req.params.id);
    if (!convo) return res.status(404).json({ error: "Conversation introuvable" });

    convo.messages.push({ role: req.body.role, content: req.body.content });
    await convo.save();
    console.log(`💬 Nouveau message ajouté à ${convo.title}`);
    res.json(convo);
  } catch (err) {
    console.error("❌ Erreur ajout message :", err);
    res.status(500).json({ error: err.message });
  }
});

// 🔹 6. Récupérer les messages d'une conversation
app.get("/conversations/:id/messages", async (req, res) => {
  try {
    const convo = await Conversation.findById(req.params.id);
    if (!convo) return res.status(404).json({ error: "Conversation introuvable" });

    console.log(`📨 Messages récupérés pour ${convo.title} (${convo.messages.length})`);
    res.json(convo.messages);
  } catch (err) {
    console.error("❌ Erreur récupération messages :", err);
    res.status(500).json({ error: err.message });
  }
});

// ================== START SERVER ==================
app.listen(PORT, () => {
  console.log(`🚀 Backend running at http://127.0.0.1:${PORT}`);
});
